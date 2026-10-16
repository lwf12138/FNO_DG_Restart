#include "cole_cole.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace colecole;
using Clock = std::chrono::steady_clock;

static double seconds(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now()-begin).count();
}

static Real evaluate(const Vec& coeff, const Result& r, int cell, Real x) {
    const Real xi=(x-r.mesh.xc[cell])/r.mesh.J[cell];
    Vec basis{1,xi};
    for(int n=1;n<r.cfg.k;++n)
        basis.push_back(xi*basis[n]-Real(n*n)/Real(4*n*n-1)*basis[n-1]);
    Real value=0;
    for(int m=0;m<=r.cfg.k;++m) value+=coeff[cell*(r.cfg.k+1)+m]*basis[m];
    return value;
}

int main(int argc, char** argv) { try {
    const uint16_t endian=1;
    if (*reinterpret_cast<const unsigned char*>(&endian) != 1 || sizeof(Real)!=8)
        throw std::runtime_error("adapter requires little-endian float64 host");
    Config cfg=default_config();
    cfg.sourceFree=true; cfg.useInitialModalCoefficients=true;
    cfg.N=64; cfg.k=2; cfg.T=1; cfg.dt=.0005;
    cfg.initialMethod="l2"; cfg.computeCorrectionErrors=false;
    cfg.verbose=false; cfg.rkDiagnostic=true;
    int points=65; std::string output, initialPath;
    for(int i=1;i<argc;i+=2) {
        if(i+1>=argc) throw std::runtime_error("missing argument value");
        const std::string key=argv[i], v=argv[i+1];
        if(key=="--alpha") cfg.alpha=std::stod(v);
        else if(key=="--tau") cfg.tau0=std::stod(v);
        else if(key=="--N") cfg.N=std::stoi(v);
        else if(key=="--k") cfg.k=std::stoi(v);
        else if(key=="--T") cfg.T=std::stod(v);
        else if(key=="--dt") cfg.dt=std::stod(v);
        else if(key=="--points") points=std::stoi(v);
        else if(key=="--initial-coefficients") initialPath=v;
        else if(key=="--cache") cfg.rkCqCacheDir=v;
        else if(key=="--output") output=v;
        else throw std::runtime_error("unknown argument: "+key);
    }
    if(output.empty()||initialPath.empty()||cfg.rkCqCacheDir.empty()||
       cfg.alpha<=0||cfg.alpha>=1||cfg.tau0<=0||!std::isfinite(cfg.alpha)||
       !std::isfinite(cfg.tau0)||cfg.N<1||cfg.k!=2||points!=65||cfg.T!=1||
       cfg.dt<=0||!std::isfinite(cfg.dt)) throw std::runtime_error("invalid acceptance configuration");
    const double requestedNt=cfg.T/cfg.dt;
    if(std::abs(requestedNt-std::round(requestedNt))>1e-9)
        throw std::runtime_error("T must be an integer multiple of requested dt");
    cfg.queryTimes={cfg.T}; // Only endpoint output, never an exact/manufactured error call.
    cfg.resultsDir=output;
    std::filesystem::create_directories(output);
    const auto readStart=Clock::now();
    const std::size_t count=static_cast<std::size_t>(3*cfg.N*(cfg.k+1));
    if(std::filesystem::file_size(initialPath)!=count*sizeof(Real))
        throw std::runtime_error("initial coefficient byte length mismatch");
    cfg.initialModalCoefficients.resize(count);
    std::ifstream input(initialPath,std::ios::binary);
    input.read(reinterpret_cast<char*>(cfg.initialModalCoefficients.data()),
               static_cast<std::streamsize>(count*sizeof(Real)));
    if(!input) throw std::runtime_error("could not read initial coefficients");
    for(std::size_t i=0;i<count;++i) {
        if(!std::isfinite(cfg.initialModalCoefficients[i]))
            throw std::runtime_error("non-finite initial coefficient");
        if(i>=2*count/3&&cfg.initialModalCoefficients[i]!=0)
            throw std::runtime_error("P0 must be exactly zero");
    }
    const double initialReadSeconds=seconds(readStart);
    const auto solveStart=Clock::now();
    Result r=solve_cole_cole_dg(cfg);
    const double solveSeconds=seconds(solveStart);
    if(r.snapshotTimes.size()!=1||std::abs(r.snapshotTimes[0]-cfg.T)>5e-14||
       std::abs(r.dt-cfg.dt)>5e-16||r.Nt!=static_cast<int>(std::llround(requestedNt)))
        throw std::runtime_error("endpoint or timestep mismatch");
    const Vec* v[]={&r.snapshotsE[0],&r.snapshotsH[0],&r.snapshotsP[0]};
    for(const Vec* field:v) for(Real value:*field)
        if(!std::isfinite(value)) throw std::runtime_error("non-finite evolved coefficient");
    const auto sampleStart=Clock::now();
    Matrix sampled(3,points); Vec xgrid(points);
    for(int j=0;j<points;++j) {
        const Real x=cfg.xa+(cfg.xb-cfg.xa)*Real(j)/Real(points-1); xgrid[j]=x;
        const int numerator=j*cfg.N, quotient=numerator/(points-1);
        const int right=std::min(cfg.N-1,quotient);
        int left=right;
        if(numerator%(points-1)==0&&j>0) left=std::max(0,quotient-1);
        for(int f=0;f<3;++f) sampled(f,j)=(evaluate(*v[f],r,left,x)+evaluate(*v[f],r,right,x))/2;
    }
    const double samplingSeconds=seconds(sampleStart);
    const auto outputStart=Clock::now();
    std::ofstream fields(output+"/fields.csv"), initial(output+"/initial_coefficients.f64",std::ios::binary);
    std::ofstream final(output+"/final_coefficients.f64",std::ios::binary);
    if(!fields||!initial||!final) throw std::runtime_error("cannot open acceptance outputs");
    fields<<std::setprecision(17)<<"time,x,E,H,P\n";
    for(int j=0;j<points;++j)
        fields<<cfg.T<<","<<xgrid[j]<<","<<sampled(0,j)<<","<<sampled(1,j)<<","<<sampled(2,j)<<"\n";
    initial.write(reinterpret_cast<const char*>(cfg.initialModalCoefficients.data()),
                  static_cast<std::streamsize>(count*sizeof(Real)));
    for(const Vec* field:v)
        final.write(reinterpret_cast<const char*>(field->data()),
                    static_cast<std::streamsize>(field->size()*sizeof(Real)));
    fields.close(); initial.close(); final.close();
    if(!fields||!initial||!final) throw std::runtime_error("output write failed");
    const double outputSeconds=seconds(outputStart);
    std::ofstream meta(output+"/metadata.json"); meta<<std::setprecision(17);
    meta<<"{\"alpha\":"<<cfg.alpha<<",\"tau\":"<<cfg.tau0<<",\"N\":"<<cfg.N
        <<",\"k\":"<<cfg.k<<",\"T\":"<<cfg.T<<",\"dt\":"<<r.dt<<",\"Nt\":"<<r.Nt
        <<",\"points\":"<<points<<",\"times\":["<<r.snapshotTimes[0]<<"]"
        <<",\"query_times\":["<<cfg.T<<"],\"epsilon_b\":2,\"mu\":1,\"delta_epsilon\":1,\"L\":"<<cfg.xb-cfg.xa
        <<",\"precision\":\"float64\",\"problem\":\"source-free\",\"P0\":0,\"initial_history\":\"zero\""
        <<",\"initial_method\":\"external full DG L2 coefficients\",\"basis\":\"monic Legendre\""
        <<",\"coefficient_shape\":[3,"<<cfg.N<<","<<cfg.k+1<<"],\"coefficient_dtype\":\"little-endian float64\""
        <<",\"sampling\":\"internal interfaces: arithmetic mean of two traces; physical endpoints: interior trace\""
        <<",\"history\":\""<<r.rkHistoryAlgorithm<<"\",\"complete_history\":true"
        <<",\"history_min_block\":"<<r.rkHistoryFftMinBlock
        <<",\"cq_target_eps\":"<<cfg.rkCqTargetEps<<",\"cq_fft_length\":"<<r.rkCqInfo.fftLength
        <<",\"cq_max_imaginary_weight\":"<<r.rkCqInfo.maxImaginaryWeightPart
        <<",\"cache_hit\":"<<(r.rkCqInfo.cacheHit?"true":"false")
        <<",\"cache_scope\":\"CQ weights on disk only; block LU rebuilt each process\""
        <<",\"threads\":1,\"spatial_quadrature_per_cell\":"<<cfg.NumGLP
        <<",\"initial_coefficient_read_seconds\":"<<initialReadSeconds
        <<",\"initial_projection_seconds\":null,\"initial_projection_location\":\"Python wrapper\""
        <<",\"solve_seconds\":"<<solveSeconds<<",\"sampling_seconds\":"<<samplingSeconds
        <<",\"output_seconds\":"<<outputSeconds<<",\"output_timing_excludes\":\"metadata and stdout\""
        <<",\"cq_weight_seconds\":"<<r.rkWeightWallSeconds<<",\"time_loop_seconds\":"<<r.rkTimeLoopWallSeconds
        <<",\"history_seconds\":"<<r.rkHistoryWallSeconds
        <<",\"history_max_imaginary_ratio\":"<<r.rkHistoryMaxImaginaryRatio
        <<",\"history_max_direct_difference\":"<<r.rkHistoryMaxDirectDifference
        <<",\"max_endpoint_difference\":"<<r.maxEndpointDiff<<"}\n";
    meta.close(); if(!meta) throw std::runtime_error("metadata write failed");
    std::cout<<"Completed source-free acceptance DG: solve_seconds="<<solveSeconds<<"\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; } }

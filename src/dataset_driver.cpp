#include "cole_cole.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace colecole;
Vec parse(const std::string& value) {
    Vec result; std::stringstream stream(value); std::string item;
    while(std::getline(stream,item,',')) result.push_back(std::stod(item));
    return result;
}
Real evaluate(const Vec& coeff,const Result& r,int cell,Real x) {
    Real xi=(x-r.mesh.xc[cell])/r.mesh.J[cell];
    Vec basis{1,xi};
    for(int n=1;n<r.cfg.k;++n) basis.push_back(xi*basis[n]-Real(n*n)/Real(4*n*n-1)*basis[n-1]);
    Real value=0;
    for(int m=0;m<=r.cfg.k;++m) value+=coeff[cell*(r.cfg.k+1)+m]*basis[m];
    return value;
}
int main(int argc,char**argv) { try {
    Config cfg=default_config(); cfg.sourceFree=true;
    cfg.N=64; cfg.k=2; cfg.T=1; cfg.dt=0.0005;
    cfg.initialMethod="l2"; cfg.computeCorrectionErrors=false; cfg.verbose=false; cfg.rkDiagnostic=true;
    cfg.queryTimes={0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1};
    int points=65; std::string output;
    for(int i=1;i<argc;i+=2) {
        if(i+1>=argc)throw std::runtime_error("missing argument value");
        std::string key=argv[i],v=argv[i+1];
        if(key=="--alpha")cfg.alpha=std::stod(v); else if(key=="--tau")cfg.tau0=std::stod(v);
        else if(key=="--a")cfg.initialElectricCoeffs=parse(v); else if(key=="--b")cfg.initialMagneticCoeffs=parse(v);
        else if(key=="--N")cfg.N=std::stoi(v); else if(key=="--k")cfg.k=std::stoi(v);
        else if(key=="--dt")cfg.dt=std::stod(v); else if(key=="--T")cfg.T=std::stod(v);
        else if(key=="--times")cfg.queryTimes=parse(v); else if(key=="--points")points=std::stoi(v);
        else if(key=="--cache")cfg.rkCqCacheDir=v; else if(key=="--output")output=v;
        else if(key=="--verbose")cfg.verbose=std::stoi(v)!=0;
        else throw std::runtime_error("unknown argument: "+key);
    }
    if(output.empty()||cfg.initialElectricCoeffs.size()!=3||cfg.initialMagneticCoeffs.size()!=3||
       cfg.alpha<=0||cfg.alpha>=1||cfg.tau0<=0||points<2||cfg.N<1||cfg.k<1||cfg.dt<=0||cfg.T<=0)
        throw std::runtime_error("invalid configuration");
    std::filesystem::create_directories(output); cfg.resultsDir=output;
    auto start=std::chrono::steady_clock::now();
    Result r=solve_cole_cole_dg(cfg);
    double solveSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    if(r.snapshotTimes.size()!=cfg.queryTimes.size())throw std::runtime_error("snapshot count mismatch");
    std::ofstream binary(output+"/coefficients.f64",std::ios::binary),fields(output+"/fields.csv");
    if(!binary||!fields)throw std::runtime_error("cannot open output");
    fields<<std::setprecision(17)<<"time,x,E,H,P\n";
    for(size_t ti=0;ti<r.snapshotTimes.size();++ti) {
        if(abs(r.snapshotTimes[ti]-cfg.queryTimes[ti])>5e-14)throw std::runtime_error("snapshot time mismatch");
        const Vec* v[]={&r.snapshotsE[ti],&r.snapshotsH[ti],&r.snapshotsP[ti]};
        for(int f=0;f<3;++f) {
            for(Real value:*v[f]) if(!std::isfinite(value))throw std::runtime_error("nonfinite DG coefficient");
            binary.write(reinterpret_cast<const char*>(v[f]->data()),v[f]->size()*sizeof(Real));
        }
        for(int j=0;j<points;++j) {
            Real x=cfg.xa+(cfg.xb-cfg.xa)*Real(j)/Real(points-1);
            int numerator=j*cfg.N,quotient=numerator/(points-1);
            int right=std::min(cfg.N-1,quotient),left=right;
            if(numerator%(points-1)==0&&j>0)left=std::max(0,quotient-1);
            fields<<r.snapshotTimes[ti]<<","<<x;
            for(int f=0;f<3;++f) fields<<","<<(evaluate(*v[f],r,left,x)+evaluate(*v[f],r,right,x))/2;
            fields<<"\n";
        }
    }
    binary.close(); fields.close();
    if(!binary||!fields)throw std::runtime_error("output write failure");
    std::ofstream meta(output+"/metadata.json"); meta<<std::setprecision(17);
    auto array=[&](const Vec& values) {meta<<"[";for(size_t i=0;i<values.size();++i){if(i)meta<<",";meta<<values[i];}meta<<"]";};
    meta<<"{\"alpha\":"<<cfg.alpha<<",\"tau\":"<<cfg.tau0<<",\"N\":"<<cfg.N<<",\"k\":"<<cfg.k
        <<",\"T\":"<<cfg.T<<",\"dt\":"<<r.dt<<",\"Nt\":"<<r.Nt
        <<",\"precision\":\"float64\",\"problem\":\"source-free\",\"initial_method\":\"l2\",\"P0\":0"
        <<",\"epsilon_b\":2,\"mu\":1,\"delta_epsilon\":1,\"L\":"<<cfg.xb-cfg.xa
        <<",\"history\":\""<<r.rkHistoryAlgorithm<<"\",\"history_min_block\":"<<r.rkHistoryFftMinBlock
        <<",\"cq_target_eps\":"<<cfg.rkCqTargetEps<<",\"cq_fft_length\":"<<r.rkCqInfo.fftLength
        <<",\"cq_max_imaginary_weight\":"<<r.rkCqInfo.maxImaginaryWeightPart
        <<",\"cache_hit\":"<<(r.rkCqInfo.cacheHit?"true":"false")
        <<",\"solve_seconds\":"<<solveSeconds<<",\"history_max_imaginary_ratio\":"<<r.rkHistoryMaxImaginaryRatio
        <<",\"history_max_direct_difference\":"<<r.rkHistoryMaxDirectDifference
        <<",\"max_endpoint_difference\":"<<r.maxEndpointDiff<<",\"a\":"; array(cfg.initialElectricCoeffs);
    meta<<",\"b\":";array(cfg.initialMagneticCoeffs);meta<<",\"times\":";array(r.snapshotTimes);
    meta<<",\"coefficient_shape\":["<<r.snapshotTimes.size()<<",3,"<<cfg.N<<","<<cfg.k+1<<"]"
        <<",\"coefficient_dtype\":\"little-endian float64\",\"points\":"<<points<<"}\n";
    std::cout<<"Completed alpha="<<cfg.alpha<<" tau="<<cfg.tau0<<" N="<<cfg.N<<" k="<<cfg.k<<" solve_seconds="<<solveSeconds<<"\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}

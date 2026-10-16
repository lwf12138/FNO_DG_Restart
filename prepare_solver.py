"""Reproducible source-free initial-condition interface around the verified double kernel."""
from pathlib import Path
import difflib
import hashlib
import json
ROOT = Path(__file__).resolve().parent
h = (ROOT/'vendor/base_double/include/cole_cole.hpp').read_text()
c = (ROOT/'vendor/base_double/src/cole_cole.cpp').read_text()
old_h, old_c = h, c
h = h.replace('struct Config {', 'struct Config {\n    bool sourceFree = false;\n    Vec initialElectricCoeffs, initialMagneticCoeffs;', 1)
needle = '''    Matrix Estar, Hstar;
    project_star_maxwell(mesh, basis, quad, cfg, exact, t, 0, Estar, Hstar);'''
replacement = '''    if (cfg.sourceFree) {
        if (cfg.initialMethod != "l2" || t != 0)
            throw std::runtime_error("source-free initial data require L2 projection at t=0");
        const Real length = cfg.xb-cfg.xa;
        auto electric = [&](const Real& x) {
            Real value=0;
            for (std::size_t m=0;m<cfg.initialElectricCoeffs.size();++m)
                value += cfg.initialElectricCoeffs[m]*sin(Real(m+1)*pi()*(x-cfg.xa)/length);
            return value;
        };
        auto magnetic = [&](const Real& x) {
            Real value=0;
            for (std::size_t m=0;m<cfg.initialMagneticCoeffs.size();++m)
                value += cfg.initialMagneticCoeffs[m]*cos(Real(m+1)*pi()*(x-cfg.xa)/length);
            return value;
        };
        InitialModal initial;
        initial.E = project_l2_func(mesh,basis,quad,electric);
        initial.H = project_l2_func(mesh,basis,quad,magnetic);
        initial.P = Matrix(mesh.N,basis.dim);
        return initial;
    }
    Matrix Estar, Hstar;
    project_star_maxwell(mesh, basis, quad, cfg, exact, t, 0, Estar, Hstar);'''
start = c.index('InitialModal initial_modal(')
end = c.index('\nnamespace {', start)
initial_function = c[start:end]
assert initial_function.count(needle) == 1
c = c[:start] + initial_function.replace(needle, replacement, 1) + c[end:]
needle = '''        for (int stage = 0; stage < s; ++stage) {
            Real ts = tn + rk.c[stage]*dt;
            FEstage[stage] = cfg.epsilon0*cfg.epsilon_inf*exact.gd(ts, 1)'''
replacement = '''        for (int stage = 0; stage < s; ++stage) {
            if (cfg.sourceFree) continue; // Both source vectors remain exactly zero.
            Real ts = tn + rk.c[stage]*dt;
            FEstage[stage] = cfg.epsilon0*cfg.epsilon_inf*exact.gd(ts, 1)'''
assert c.count(needle) == 1
c = c.replace(needle,replacement)
for rel,old,new in [('include/cole_cole.hpp',old_h,h),('src/cole_cole.cpp',old_c,c)]:
    (ROOT/rel).write_text(new)
    (ROOT/'checks'/(Path(rel).name+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='base_double/'+rel,tofile=rel)))
print('Prepared initial-function interface and zero-source branch; spatial/RK/CQ updates unchanged.')

import numpy as np
import cbsyst as cb
from tqdm.notebook import tqdm

from .helpers import calc_rho_f, calc_rho_p, calc_Rdiss, calc_Rremin, calc_wsink, sphere_vol_from_radius, sphere_radius_from_vol, day_2_seconds

def sinking_particles(Fc=0.05, r0=250, k_diss=4.5e-3, p_lifetime=2.5, tmax_days=10, tsteps=800, N=2000, T=5, S=35):

    t = np.logspace(-3, np.log10(tmax_days * day_2_seconds), tsteps + 2)
    log_dt = t[1] - t[0]
    # t = np.linspace(0, tmax_days * day_2_seconds, tsteps + 2)

    Ca = 10.2e-3

    ### organic matter
    rho_org = 1035.
    MW_org = 106 * 12 + 175 + 42 * 16 + 16 * 14 + 31 # 106C 175H 42O 16N P

    # change in DIC and TA associated with 1M of organic matter remineralisation
    n_DIC_org = 106
    n_TA_org = -18
    n_O2_org = -150

    # organic matter particle life time
    p_lifetime = p_lifetime * day_2_seconds
    R_remin = 1 / p_lifetime

    rho_carb = 2710.  # CaCO3 density (calcte = 2710, aragonite = 2940)
    MW_carb = 100.  # CaCO3 molecular weight

    # change in DIC and TA per mole of CaCO3 dissolution
    n_DIC_carb = 1
    n_TA_carb = 2

    # dissolution parameters
    n_diss = 2.7

    # Vmix = 0  # kgSW s-1

    init = {
        'z': 0,
        'R_z': 0,
        'r': r0 * 1e-6,
        'P': 0,
        'DIC': 2050,
        'TA': 2200,
        'O2': 200,
        'Fc': Fc,
        'rho_p': calc_rho_p(Fc, rho_org=rho_org, rho_carb=rho_carb)
    }
    Vp = sphere_vol_from_radius(init['r'])
    init['POC'] = n_DIC_org * N * 1e6 * Vp * (1 - init['Fc']) * rho_org * 1000 / MW_org  # umol kg-1
    init['PIC'] = n_DIC_carb * N * 1e6 * Vp * init['Fc'] * rho_carb * 1000 / MW_carb  # umol kg-1
    init['PC'] = init['PIC'] + init['POC']

    sw = cb.Csys(DIC=init['DIC'], TA=init['TA'], T_in=T, S_in=S)
    init['CO3'] = sw.CO3
    init['Omega'] = Ca * sw.CO3 * 1e-6 / sw.Ks.KspA
    init['R_diss'] = 0
    init['R_remin'] = 0

    m = {}
    for k, v in init.items():
        m[k] = np.zeros(t.shape)
        m[k][:] = v

    for i in tqdm(range(1, len(t) - 1), total=len(t) - 2):
        ii = i - 1
        dt = t[i] - t[ii]

        # sinking rate    
        dz_dt = calc_wsink(m['r'][ii], m['rho_p'][ii], calc_rho_f(m['z'][ii]))  # m s-1
        m['z'][i] = m['z'][ii] + dz_dt * dt
        m['P'][i] = m['z'][i] / 10.  # pressure in bar

        m['R_z'][i] = dz_dt

        # calculate volumes
        i_V = sphere_vol_from_radius(m['r'][ii])
        i_Vo = i_V * (1 - m['Fc'][ii])
        i_Vc = i_V * m['Fc'][ii]

        # remineralisation 
        dVo_dt = calc_Rremin(R_remin, m['O2'][i], K_M=1) * i_Vo  # m3 organic matter s-1
        # dissolution
        dVc_dt = i_Vc * calc_Rdiss(m['Omega'][ii], k=k_diss, n=n_diss)  # m3 CaCO3 s-1        

        # shrink particle
        new_Vo = max(0, i_Vo - dVo_dt * dt)
        new_Vc = max(0, i_Vc - dVc_dt * dt)
        new_V = new_Vo + new_Vc
        if new_V == 0:
            m['Fc'][i] = np.nan
        else:
            m['Fc'][i] = new_Vc / new_V
        m['r'][i] = sphere_radius_from_vol(new_V)

        m['POC'][i] = n_DIC_org * N * 1e6 * new_Vo * rho_org * 1000 / MW_org
        m['PIC'][i] = n_DIC_carb * N * 1e6 * new_Vc * rho_carb * 1000 / MW_carb  # umol kg-1
        m['PC'][i] = m['PIC'][i] + m['POC'][i]
        
        # calculate moles released per kg of water
        dMo_dt = N * 1e6 * dVo_dt * rho_org * 1000 / MW_org  # umol organic matter kgSW-1 s-1
        dMc_dt = N * 1e6 * dVc_dt * rho_carb * 1000 / MW_carb # umol CaCO3 kgSW-1 s-1

        m['R_remin'][ii] = dMo_dt
        m['R_diss'][ii] = dMc_dt
        
        # if moved greater than 10 cm it's in new water
        dz = abs(dz_dt * dt)
        if dz > 0.1:
            mi = i
            mf = 1
        else:
            mi = ii
            mf = dz / 0.1

        m['DIC'][i] = (
            (m['DIC'][i] * mf + m['DIC'][mi] * (1-mf)) +  # background state
            n_DIC_org * dMo_dt * dt +  # remineralisation
            n_DIC_carb * dMc_dt * dt  # dissolution
        )
        m['TA'][i] = (
            m['TA'][i] * mf + m['TA'][mi] * (1-mf) +  # background state
            n_TA_org * dMo_dt * dt +  # remineralisation
            n_TA_carb * dMc_dt * dt  # dissolution
        )
        m['O2'][i] = max(0, (
            m['O2'][i] * mf + m['O2'][mi] * (1-mf) +   # background state
            n_O2_org * dMo_dt * dt  # remineralisation
        ))

        # mixing
        # for p in ['O2', 'TA', 'DIC']:
        #     m[p][i] += Vmix * dt * ((m[p][i] - m[p][ii]) + (m[p][i] - m[p][i+1]))

        # density for next step
        m['rho_p'][i] = calc_rho_p(m['Fc'][i], rho_org=rho_org, rho_carb=rho_carb)

        isw = cb.Csys(DIC=m['DIC'][i], TA=m['TA'][i], S_in=S, T_in=T, P_in=m['P'][i])
        m['CO3'][i] = isw.CO3
        m['Omega'][i] = Ca * isw.CO3 * 1e-6 / isw.Ks.KspA

    for k in m.keys():
        m[k][0] = np.nan
        m[k][-1] = np.nan
    
    m['t'] = t
    m['rho_sw'] = calc_rho_f(m['z'])
    return m
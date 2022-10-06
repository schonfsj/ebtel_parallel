;;
;;Program to read text files from ebtelplusplus output and save them in IDL
;;so GX Simulator can use them
;;
;;filename  : The base (without extensions) name of the output ebtelplusplus
;;            files.
;;file_dir  : The absolute or relative path in which to search for 'filename'.
;;
;;

pro ebtelplusplus_read, filename, file_dir=file_dir
  test = 0
  IF not keyword_set(filename) THEN BEGIN
    prompt,filename,'Input filename base to convert:'
  ENDIF

  IF not keyword_set(file_dir) THEN file_dir = ''
  files = file_search(file_dir+filename+'*')

  IF not isa(files[0],/string) THEN BEGIN
    print,'No files found with specified identifier: '+filename
    stop
  ENDIF

  cfg_pos = where(strmatch(files, filename+'.cfg.xml'))
  IF cfg_pos ne -1 THEN BEGIN
    cfg_struct = read_xml((files[cfg_pos])[0])
    cfg = cfg_struct.root
  ENDIF ELSE BEGIN
    print,'ebtelplusplus_read ERROR: no "*.cfg.xml" file found. Stopping.'
    stop
  ENDELSE

  params_pos = where(strmatch(files, filename+'_params.txt'))
  IF params_pos ne -1 THEN BEGIN
    readcol,files[params_pos], $
            hhp,llp,lrun1,qrun1,trun1,format='L,L,F,F,F',skipline=1
  ENDIF ELSE BEGIN
    print,'ebtelplusplus_read ERROR: no "*_params.txt" file found. Stopping.'
    stop
  ENDELSE

  dems_pos = where(strmatch(files, filename+'_dems.txt'))
  IF params_pos ne -1 THEN BEGIN
    readcol,files[dems_pos], $
            hhd,lld,logt1,tr1,cor1,ddmtr1,ddmcor1, $
            format='L,L,F,F,F,F,F',skipline=1
  ENDIF ELSE BEGIN
    print,'ebtelplusplus_read ERROR: no "*_dems.txt" file found. Stopping.'
    stop
  ENDELSE

  ;;Parse ll and hh arrays
  lmin = min(llp,max=lmax)
  hmin = min(hhp,max=hmax)

  ;;Parse logt1 array
  logtdem = logt1[0]
  lgtmax = max(logt1,/nan)
  i=1
  WHILE logtdem[-1] ne lgtmax DO BEGIN
    logtdem = [logtdem,logt1[i]]
    i++
  ENDWHILE
  lgt = reform(rebin(indgen(i),i,(hmax-hmin+1)*(lmax-lmin+1)),n_elements(logt1))

  ;;Create desired arrays
  lrun = fltarr(hmax-hmin+1,lmax-lmin+1)
  qrun = lrun
  trun = lrun
  dem_tr_run = fltarr(n_elements(logtdem),hmax-hmin+1,lmax-lmin+1)
  dem_cor_run = dem_tr_run
  ddm_tr_run = dem_tr_run
  ddm_cor_run = dem_tr_run

  ;;Assign parameter arrays
  lrun[hhp-hmin,llp-lmin] = lrun1
  qrun[hhp-hmin,llp-lmin] = qrun1
  trun[hhp-hmin,llp-lmin] = trun1

  ;;Assign dem arrays
  dem_tr_run[lgt,hhd-hmin,lld-lmin] = tr1
  dem_cor_run[lgt,hhd-hmin,lld-lmin] = cor1

  ;;Assign ddm arrays
  ddm_tr_run[lgt,hhd-hmin,lld-lmin] = ddmtr1
  ddm_cor_run[lgt,hhd-hmin,lld-lmin] = ddmcor1

  save,filename=file_dir+'gx_sav/'+filename+'.sav', $
      cfg,logtdem,lrun,qrun,trun,dem_tr_run,dem_cor_run,ddm_tr_run,ddm_cor_run
  print, 'Save file written to ' +file_dir+'gx_sav/'+filename+'.sav'

  IF strpos(filename,'test') ne -1 THEN BEGIN ;filename with 'test' in it
    logtdem_t = logtdem
    lrun_t = lrun
    qrun_t = qrun
    trun_t = trun
    dem_tr_run_t = dem_tr_run
    dem_cor_run_t = dem_cor_run

    ;; Must point to the SSW GX simulator installation for comparison
    restore,'/ssw/packages/gx_simulator/userslib/aia/ebtel.sav'
    l = 0
    h = 0

    check_length = [1e8, 1e10, 1e8,1e10]
    check_heat = [1e-1, 1e-2, 1e-5, 1e-6]

    heat_max = max([max(qrun),max(qrun_t)])
    heat_min = min([min(qrun),min(qrun_t)])

    ideal_bg = 6.3e12 / (lrun_t[0,*]^2)
    ideal_bg >= 1e-7

    !P.Multi = [0, 2, 1]
    window,0,xsize=1300,ysize=500

    cgplot, lrun, qrun, psym=3, /xlog, /ylog, yrange=[heat_min, heat_max], $
      xtitle='L [cm]', ytitle='<Q> [erg cm^-3 s^-1]'
    cgplot, /over, lrun_t, qrun_t, psym=3, color='red'
    cgplot, /over, lrun_t[0,*], ideal_bg, color='blue'
    cgplot, /over, check_length, check_heat, psym=2, color='purple'

    !P.charsize = 1.5
    !P.Multi = [6, 4, 2]
    FOR ii=0,size(check_length,/n_elements)-1 DO BEGIN
      if ii eq 2 then !P.Multi = [2, 4, 2]
      length_ind_test = value_locate(lrun_t[0,*], check_length[ii])
      length_ind = value_locate(lrun[0,*], check_length[ii])
      heat_ind_test = value_locate(qrun_t[*,length_ind_test], check_heat[ii])
      heat_ind = value_locate(qrun[*,length_ind], check_heat[ii])

      print,length_ind,heat_ind, (1.e14)/(((10.*10.^6.)+2.*check_length[ii])^2.)
      dem_max = max(dem_cor_run[*,heat_ind,length_ind])

      cgplot, logtdem, dem_cor_run[*,heat_ind,length_ind], $
        yrange=[dem_max*1e-5,dem_max*10], /ylog, $
        xtitle = 'Log(T) [K]', ytitle = 'DEM [cm$\up-5$ K$\up-1$]', $
        title = 'L=' + string(check_length[ii],format='(e8.1)') $
                + ', Q=' + string(check_heat[ii], format='(e8.1)')
      cgplot,/over,logtdem_t, dem_cor_run_t[*,heat_ind_test,length_ind_test], $
        color='red'
    ENDFOR

    write_png, 'gx_sav/'+filename+'.png', tvrd(/true)
    stop
  ENDIF

end

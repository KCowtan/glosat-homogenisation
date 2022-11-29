"""
Program to determine station uncertainties.

Arguments:
 -i=<filename> : specific input pkl file (default ../DATA/df_temp.pkl)
 -o=<filename> : specific output pkl file (default df_temp_expect.pkl) 
 -years=<year>,<year> : years for calculation (default 1780,2020)
"""
import sys, math, numpy, pandas, ruptures, glosat_homogenization
import statsmodels.api as sm
import matplotlib.pyplot as plt



def anomaly_diff2( x ):
  """
  Calculate mean squared difference with removal of monthly difference
  """
  s = 0.0
  for m in range(12):
    y = x[m::12]
    s += numpy.sum(numpy.square(y-numpy.mean(y)))
  return s/(len(x)-12)


# MAIN PROGRAM
def main():
  # command line arguments
  year0,year1 = 1780,2020
  ystep = 50
  stationfilter = None
  ifile = None
  ofile = None

  for arg in sys.argv[1:]:
    if arg.split("=")[0] == "-i":        # input file
      ifile = arg.split("=")[1]
    if arg.split("=")[0] == "-o":        # output file
      ofile = arg.split("=")[1]
    if arg.split("=")[0] == "-years":    # year calc
      year0,year1 = [int(x) for x in arg.split("=")[1].split(",")]
    if arg.split("=")[0] == "-step":     # year step 
      ystep = int(arg.split("=")[1])
    if arg.split("=")[0] == "-filter":   # station selection
      stationfilter = arg.split("=")[1]


  # other defaults
  if ifile == None: ifile = "../DATA/df_temp.pkl"
  if ofile == None: ofile = "df_temp_errors.pkl"

  # years and months
  years  = numpy.arange(year0,year1+1)
  months = numpy.arange(1,13)

  # read station data
  dsrc = pandas.read_pickle( ifile, compression='bz2' )

  print(numpy.unique(dsrc["stationcode"]))
  # filter if required
  if stationfilter:
    stnmask = dsrc["stationcode"].str.startswith(stationfilter)
  else:
    stnmask = numpy.full( [dsrc.shape[0]], True )
  stnmask = numpy.logical_and( stnmask, dsrc["stationlat"].notna() )
  stnmask = numpy.logical_and( stnmask, dsrc["stationlon"].notna() )
  dflt = dsrc[stnmask]

  # extract 1 row per station
  dsub = dflt.drop_duplicates( subset="stationcode" )
  print(dsub.columns)
  print(dsub.describe())

  # extract column info
  codes = dsub.loc[:,"stationcode"].values
  names = dsub.loc[:,"stationname"].values
  ctrys = dsub.loc[:,"stationcountry"].values
  lats  = dsub.loc[:,"stationlat"].values
  lons  = dsub.loc[:,"stationlon"].values
  dists = glosat_homogenization.prepare_dists( lats, lons )

  # and full data
  dcodes = dflt.loc[:,"stationcode"].values
  dyears = dflt.loc[:,"year"].values
  dtemps = dflt.loc[:,["1","2","3","4","5","6","7","8","9","10","11","12"]].values

  # make data table
  nstn = len(codes)
  nmon = 12*len(years)
  data  = numpy.full( [nmon,nstn], numpy.nan )
  dates = numpy.full( [nmon,2], 0.0 )
  for j in range(nmon):
    y = years[j//12]
    m = months[j%12]
    dates[j,0] = y
    dates[j,1] = m

  # fill table by station and month
  for r in range(dcodes.shape[0]):
    if year0 <= dyears[r] <= year1:
      i = 12*(dyears[r]-year0)
      j = numpy.argmax( dcodes[r] == codes )
      for m in range(12):
        data[i+m,j] = dtemps[r,m]

  # loop over time windows
  resultyears, resulterrs = [], []
  for baseyear in range( year0, year1, ystep ):
    print("calculating ",baseyear)

    # year mask
    ymask = numpy.logical_and( dates[:,0] >= baseyear,
                               dates[:,0] <  baseyear+ystep )
    nmask = numpy.sum(ymask)

    # calculate rmsd minimums
    r2 = numpy.full( dists.shape, numpy.nan )
    nwin = 60
    m0 = numpy.count_nonzero(~numpy.isnan(data),axis=0)
    n1 = numpy.arange(0,nstn)
    m1 = m0>=2*nwin
    print(sum(m1))
    # calc best residual for each station pair
    for s1 in n1[m1]:
      n2 = numpy.arange(s1+1,nstn) # optimized selection by range
      m2 = numpy.logical_and(numpy.logical_and(dists[s1,n2]>10,dists[s1,n2]<500),m0[n2]>2*nwin)
      while sum(m2)>200: m2 = numpy.logical_and(m2,numpy.random.randint(0,2,size=m2.shape))
      print(s1,sum(m2),codes[s1],names[s1])
      for s2 in n2[m2]:
        d = data[ymask,s1] - data[ymask,s2]
        if numpy.count_nonzero(~numpy.isnan(d))>2*nwin:
          r = []
          for y in range(0,nmask-nwin,12):
            dw = d[y:y+nwin]
            # if there are missings, try and backfill from extended window
            if 0 < numpy.count_nonzero(numpy.isnan(dw)) <= 24:
              yoff = min(max(y,12),nmask-nwin-12)
              dwx = d[yoff-12:yoff+nwin+12]
              for m in range(12):
                if numpy.count_nonzero(numpy.isnan(dw[m::12]))>0:
                  dm = dwx[m::12]
                  dm = dm[~numpy.isnan(dm)]
                  if dm.shape[0] >= nwin//12: dw[m::12] = dm[:nwin//12]
            # if there are no missings, add the result
            if numpy.count_nonzero(numpy.isnan(dw)) == 0:
              r.append( anomaly_diff2( dw ) )
          if len(r) > 0:
            if 1.0e-2 < min(r) < 4.0:
              r2[s1,s2] = r2[s2,s1] = 0.5*min(r)

    # get OLS fit of residual against distance
    err = numpy.full( [nstn], numpy.nan )
    for s in range(nstn):
      m = ~numpy.isnan( r2[s,:] )
      if sum(m) >= 3:
        x = sm.add_constant(dists[s,m])
        y = numpy.sqrt(r2[s,m])
        fit = sm.OLS(y,x).fit()
        om = fit.outlier_test()[:,2]>0.05
        x1 = x[om,:]
        y1 = y[om]
        if len(y1)>=3:
          fit1 = sm.OLS(y1,x1).fit()
          err[s] = fit1.params[0]
          print(s,fit.params,fit1.params,err[s])

    # fill station errors
    merr = numpy.full(err.shape,numpy.nan)
    for s in range(nstn):
      sdist = dists[:,s]
      idist = numpy.argsort(sdist)
      esort,dsort = err[idist],sdist[idist]
      m = ~numpy.isnan(esort)
      esort = esort[m]
      dsort = dsort[m]
      if numpy.median(dsort[:25]) < 1000.0:
        merr[s] = numpy.median( esort[:25] )

    # get fill value for isolated stations
    efill = numpy.quantile(err[~numpy.isnan(err)],0.75)
    merr[numpy.isnan(merr)] = efill
    print("Error fill:",efill,numpy.count_nonzero(~numpy.isnan(err)))

    # plot map
    m = ~numpy.isnan(err)
    fig = plt.figure(figsize=[8,4.5])
    plt.subplots_adjust(left=0.10,bottom=0.03,top=0.95,right=0.95)
    s = plt.scatter(lons[m],lats[m],c=err[m],s=10,vmin=0.0,vmax=1.0,cmap=plt.cm.get_cmap('rainbow'))
    plt.colorbar(s,orientation="horizontal",aspect=40,pad=0.07)
    plt.xlim([-180,180])
    plt.ylim([ -90, 90])
    plt.savefig("stations.raw.{:4d}.png".format(baseyear))
    plt.close()

    # plot map
    m = ~numpy.isnan(merr)
    fig = plt.figure(figsize=[8,4.5])
    plt.subplots_adjust(left=0.10,bottom=0.03,top=0.95,right=0.95)
    s = plt.scatter(lons[m],lats[m],c=merr[m],s=10,vmin=0.0,vmax=1.0,cmap=plt.cm.get_cmap('rainbow'))
    plt.colorbar(s,orientation="horizontal",aspect=40,pad=0.07)
    plt.xlim([-180,180])
    plt.ylim([ -90, 90])
    plt.savefig("stations.med.{:4d}.png".format(baseyear))
    plt.close()

    # store results
    resultyears.append( baseyear )
    resulterrs.append( merr )

    """
    # plot curves
    print("plotting")
    for s in range(nstn):
      m = ~numpy.isnan( r2[s,:] )
      x = dists[s,m]
      y = numpy.sqrt(r2[s,m])
      n = numpy.sort(numpy.array([x,y]).T,axis=0)
      plt.plot(n[:,0],n[:,1],'-')
    plt.savefig("rmsd_dist.png")
    plt.close()
    """

  # fill in errors
  resultyears = numpy.array(resultyears)+0.5*ystep
  resulterrs = numpy.array(resulterrs)
  print(resultyears.shape,resulterrs.shape)

  def bspl(x):
    x1 = abs(x)
    if x1 > 1.5:
      return 0.0
    elif x1 > 0.5:
      return 0.5*(1.5-x1)**2
    else:
      return 0.75-x1**2

  def bsplineinterp(x1,x,y):
    fx1 = (x1-x[0])/(x[1]-x[0])
    ix1 = int(round(fx1))
    ix0 = max(ix1-1,0)
    ix2 = min(ix1+1,len(x)-1)
    dx = fx1-ix1
    wx0,wx1,wx2 = bspl(dx+1.0),bspl(dx),bspl(dx-1.0)
    return wx0*y[ix0]+wx1*y[ix1]+wx2*y[ix2]

  # smooth and fill in errors
  stnerrs = numpy.full( data.shape, numpy.nan )
  for s in range(nstn):
    for m in range(nmon):
      stnerrs[m,s] = bsplineinterp( dates[m,0], resultyears, resulterrs[:,s] )
    """
    plt.title( names[s] )
    plt.plot( resultyears, resulterrs[:,s], 'ko' )
    plt.plot( dates[:,0], stnerrs[:,s], 'g-' )
    plt.show()
    """

  # DATA OUTPUT
  # We need to add missing years (rows) to the source dataframe, then add
  # extra columns for the additional information
  # Make output data frame
  nyr = len(years)
  ddst = pandas.DataFrame( columns=["year","stationcode"] )
  # now populate from source data frame
  ddst["year"] = numpy.tile( years, nstn )
  ddst["stationcode"] = numpy.repeat( codes, nyr )

  # make new columns for source dataframe
  # we need to do this in two steps because updating the dataframe is very slow
  print("Making data\n",flush=True)
  svals = numpy.full( [ddst.shape[0],12], numpy.nan )
  for r in range(len(ddst)):
    year = ddst.at[ r, "year" ]
    code = ddst.at[ r, "stationcode" ]
    j = numpy.argmax( code == codes )
    if year0 <= year <= year1 and codes[j] == code:
      i = 12*(year-year0)
      svals[r,:] = stnerrs[i:i+12,j]
    if r%1000 == 0: print(" row ",r,flush=True)

  # add the new columns to the source data frame
  print("Copying data\n",flush=True)
  scols = ["sse1","sse2","sse3","sse4","sse5","sse6","sse7","sse8","sse9","sse10","sse11","sse12"]
  for m in range(12):
    ddst[scols[m]] = svals[:,m]
  print( numpy.nanmean(ddst.loc[:,"sse1"]) )

  # now join with the original data
  djoin = pandas.merge( dflt, ddst, how="outer", on=["year","stationcode"] )

  # save the data
  print("Saving data\n",flush=True)
  djoin.to_pickle( ofile, compression="bz2" )


# Main program launcher
if __name__ == '__main__':
    main()

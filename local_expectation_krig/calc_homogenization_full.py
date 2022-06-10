"""
Program to determine station norms, calculate local expectation, and
optionally homogenization using full matrix solution for the norms.

Arguments:
 -i=<filename> : specific input pkl file (default ../DATA/df_temp.pkl)
 -o=<filename> : specific output pkl file (default df_temp_expect.pkl) 
 -cycles=<ncycle> : number of cycles of changepoint detection (default 0)
 -fourier=<nfourier> : number of Fourier order per changepoint (default 0)
 -filter=<filter> : only use stations with given prefix
 -years=<year>,<year> : years for calculation (default 1780,2020)
 -bases=<year>,<year> : baseline years (default 1961,1990)

If cycles is zero (the default), then calculate local expectation only.
"""
import sys, math, numpy, pandas, ruptures, glosat_homogenization


# Change point detection using Killick 2012, Truong 2020
def changepoints( dorig, **opts ):
  """
  Change point detection using PELT (Killick 2012),
  implemented in the ruptures package (Truong 2020)
  
  Parameters:
    dorig (vector of float): original data with no seasonal cycle, e.g.
      difference between obs and local expectation. No missing values.
    opts (dictionary): additional options, including:
      "nbuf": minimum number of months between changepoints
    
  Returns:
    (list of float): list of indices of changepoints
  """  
  min_size = opts["nbuf"]
  penalty_value = 10
  algo = ruptures.KernelCPD(kernel="linear",min_size=min_size).fit(dorig)
  #algo = ruptures.Pelt(model="l2", min_size=min_size).fit(dorig)
  result = algo.predict(pen=penalty_value)
  return result[:-1]


# calculate changepoints on data with breaks
  """
  Change point detection wrapper to allow missing data and return a
  vector of flags
  
  Parameters:
    dorig (vector of float): original data with no seasonal cycle, e.g.
      difference between obs and local expectation. No missing values.
    opts (dictionary): additional options for changepoints function
    
  Returns:
    (vector of unit8): vector of station fragment flags
  """  
def changemissing( dnorm, **opts ):
  mask = ~numpy.isnan(dnorm)
  diff = dnorm[mask]
  chg = []
  if diff.shape[0] > 2*opts["nbuf"]: chg = changepoints(diff,**opts)
  index = numpy.arange( dnorm.shape[0] )[mask]
  flags = numpy.full( dnorm.shape, 0, numpy.uint8 )
  for i in chg:
    flags[index[i]:] += 1
  return flags



# MAIN PROGRAM
def main():
  # command line arguments
  year0,year1 = 1780,2020
  base0,base1 = 1961,1990
  stationfilter = None
  tor = 0.1
  nfourier = 0
  ncycle   = 10
  rebaseline = True
  ifile = None
  ofile = None

  for arg in sys.argv[1:]:
    if arg.split("=")[0] == "-i":        # input file
      ifile = arg.split("=")[1]
    if arg.split("=")[0] == "-o":        # output file
      ofile = arg.split("=")[1]
    if arg.split("=")[0] == "-years":    # year calc
      year0,year1 = [int(x) for x in arg.split("=")[1].split(",")]
    if arg.split("=")[0] == "-bases":    # year calc
      base0,base1 = [int(x) for x in arg.split("=")[1].split(",")]
    if arg.split("=")[0] == "-filter":   # station selection
      stationfilter = arg.split("=")[1]
    if arg.split("=")[0] == "-fourier":  # number of fourier orders
      nfourier = int(arg.split("=")[1])
    if arg.split("=")[0] == "-cycles":   # number of cycles of homogenization
      ncycle   = int(arg.split("=")[1])
    if arg.split("=")[0] == "-no-baseline": # disable fit baseline
      rebaseline = False

  # other defaults
  if ifile == None: ifile = "../DATA/df_temp.pkl"
  if ofile == None and stationfilter == None: ofile = "df_temp_expect.pkl"
  if ofile == None and stationfilter != None: ofile = "df_temp_expect_{:s}.pkl".format(stationfilter)

  # years and months
  years  = numpy.arange(year0,year1+1)
  months = numpy.arange(1,13)

  # read station data
  dsrc = pandas.read_pickle( ifile, compression='bz2' )

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

  # baseline period mask
  basemask = numpy.logical_and( dates[:,0] >= base0, dates[:,0] < base1 )

  # fill table by station and month
  for r in range(dcodes.shape[0]):
    if year0 <= dyears[r] <= year1:
      i = 12*(dyears[r]-year0)
      j = numpy.argmax( dcodes[r] == codes )
      for m in range(12):
        data[i+m,j] = dtemps[r,m]

  # set up covariance data
  cov = numpy.exp( -dists/900.0 )

  # set breakpoint flags - normally all empty, but we can use extreme values to mark known breaks
  # FIXME - update this once the input data contain breakpoint flags
  flags = numpy.full( data.shape, 0, numpy.uint8 )
  for s in range(nstn): flags[:,s] = data[:,s]>100.0
  flags = numpy.cumsum(flags,axis=0)
  print("Data flag removal ",numpy.nanmin(data),numpy.nanmax(data))
  data[data>100.0] -= 200.0
  print("Data flag removal ",numpy.nanmin(data),numpy.nanmax(data))

  # simple normalization for annual cycle
  # -------------------------------------
  # This step does a basic anomaly calculation to remove the bulk of the annual cycle
  # so that we don't need to deal with those parameters in the full matrix norm calcualtion.
  pnorm = glosat_homogenization.simple_norms( data )
  dnorm = data - pnorm
  print( "INIT ", numpy.nanstd(data), numpy.nanstd(pnorm), numpy.nanstd(dnorm) )

  # first full matrix normalization
  # -------------------------------
  # We create an empyty array of station breakpoint flags in order to set norms
  # using the full matrix method for complete station records.
  # For some reason starting from zero norms works better than solving from the initial data
  #norms,norme,pars,X,Q = glosat_homogenization.solve_norms( dnorm, flags, cov, tor, nfourier )
  norms,norme = numpy.full( data.shape, 0.0 ), numpy.full( data.shape, numpy.nan )
  dfull = dnorm - norms

  # calculate local expectations
  # ----------------------------
  # Now we calculate a local expectation at the location of each station using the
  # anomalies.
  dlexp,var = glosat_homogenization.local_expectation( dfull, cov, tor )
  print( "INIT ", numpy.nanstd(dnorm), numpy.nanstd(dfull), numpy.nanstd(dlexp) )

  # Iteratively find breakpoints
  # ----------------------------
  # We loop over n cycles, finding breakpoints from the difference between a station and
  # its expectation, then updateing the norms and expectations.
  for cycle in range(ncycle):
    for s in range(nstn):
      flags[:,s] = changemissing( dnorm[:,s] - dlexp[:,s], nbuf=12 )
    norms,norme,pars,X,Q = glosat_homogenization.solve_norms( dnorm, flags, cov, tor, nfourier )
    dfull = dnorm - norms
    dlexp,var = glosat_homogenization.local_expectation( dfull, cov, tor )

  print( "NORMS ", numpy.std(norms), numpy.sum(flags) )

  # calculate uncertainties
  # -----------------------
  # empirical variance estimation
  d2 = (dfull-dlexp)**2
  vmsk = numpy.logical_and( ~numpy.isnan(d2), var>0.0 )
  var *= numpy.mean( d2[vmsk] ) / numpy.mean( var[vmsk] )

  # optional fitting to baseline
  # ----------------------------
  # If required, fit the resulting expectations on the baseline window,
  # then fit the stations to the baselines.
  if rebaseline:
    dlexpm = dlexp[basemask,:]
    for m in range(12):
      dlexp[m::12,:] -= numpy.nanmean( dlexpm[m::12,:], axis=0 )

  # Output
  # ------
  # The rest of the calculation is just collecting data for output.
  # covariance data
  print(Q)
  covx,covy,covz = [],[],[]
  for i in range(len(pars)):
    for j in range(len(pars)):
      f1,s1 = pars[i]
      f2,s2 = pars[j]
      covx.append(dists[s1,s2])
      covy.append(Q[i,j])
      covz.append(numpy.count_nonzero(numpy.logical_and(flags[:,s1]==f1,flags[:,s2]==f2)))
  cov = pandas.DataFrame({"dist":covx,"cov":covy,"overlap":covz})
  print("Self:              ",numpy.mean(cov["cov"][cov["dist"]<0.5]))
  print("Other:             ",numpy.mean(cov["cov"][cov["dist"]>0.5]))
  print("Self,  overlap:    ",numpy.mean(cov["cov"][numpy.logical_and(cov["dist"]<0.5,cov["overlap"]>0.5)]))
  print("Self,  no overlap: ",numpy.mean(cov["cov"][numpy.logical_and(cov["dist"]<0.5,cov["overlap"]<0.5)]))
  print("Other, overlap:    ",numpy.mean(cov["cov"][numpy.logical_and(cov["dist"]>0.5,cov["overlap"]>0.5)]))
  print("Other, no overlap: ",numpy.mean(cov["cov"][numpy.logical_and(cov["dist"]>0.5,cov["overlap"]<0.5)]))
  cov.to_csv("cov.csv",sep=" ",index=False)

  # update station baselines to match filled data
  diff = data - norms - dlexp
  for s in range(nstn):
    for m in range(12):
      norms[m::12,s] += numpy.nanmean( diff[m::12,s] )


  # DATA OUTPUT
  # We need to add missing years (rows) to the source dataframe, then add
  # extra columns for the additional information
  # Make output data frame
  nyr = len(years)
  ddst = pandas.DataFrame( columns=["year","stationcode"] )
  # now populate from source data frame
  ddst["year"] = numpy.tile( years, nstn )
  ddst["stationcode"] = numpy.repeat( codes, nyr )

  print(dflt.shape, ddst.shape)

  # make new columns for source dataframe
  # we need to do this in two steps because updating the dataframe is very slow
  print("Making data\n",flush=True)
  nvals = numpy.full( [ddst.shape[0],12], numpy.nan )
  nvale = numpy.full( [ddst.shape[0],12], numpy.nan )
  evals = numpy.full( [ddst.shape[0],12], numpy.nan )
  svals = numpy.full( [ddst.shape[0],12], numpy.nan )
  for r in range(len(ddst)):
    year = ddst.at[ r, "year" ]
    code = ddst.at[ r, "stationcode" ]
    j = numpy.argmax( code == codes )
    if year0 <= year <= year1 and codes[j] == code:
      i = 12*(year-year0)
      nvals[r,:] = norms[i:i+12,j]
      nvale[r,:] = norme[i:i+12,j]
      evals[r,:] = dlexp[i:i+12,j]
      svals[r,:] = numpy.sqrt(var[i:i+12,j])
    if r%1000 == 0: print(" row ",r,flush=True)

  # add the new columns to the source data frame
  print("Copying data\n",flush=True)
  ncols = [ "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11", "n12"]
  ncole = ["ne1","ne2","ne3","ne4","ne5","ne6","ne7","ne8","ne9","ne10","ne11","ne12"]
  ecols = [ "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10", "e11", "e12"]
  scols = [ "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12"]
  for m in range(12):
    ddst[ncols[m]] = nvals[:,m]
  for m in range(12):
    ddst[ncole[m]] = nvale[:,m]
  for m in range(12):
    ddst[ecols[m]] = evals[:,m]
  for m in range(12):
    ddst[scols[m]] = svals[:,m]
  print( numpy.nanmean(ddst.loc[:,"n1"]), numpy.nanmean(ddst.loc[:,"ne1"]), numpy.nanmean(ddst.loc[:,"e1"]), numpy.nanmean(ddst.loc[:,"s1"]) )

  # now join with the original data
  djoin = pandas.merge( dflt, ddst, how="outer", on=["year","stationcode"] )

  # save the data
  print("Saving data\n",flush=True)
  djoin.to_pickle( ofile, compression="bz2" )



# Main program launcher
if __name__ == '__main__':
    main()


"""
Program to determine station norms, calculate local expectation, and
optionally homogenization using full matrix solution for the norms.

Arguments:
 -i=<filename> : specific input pkl file (default ../DATA/df_temp.pkl)

If cycles is zero (the default), then calculate local expectation only.
"""
import sys, math, numpy, pandas, glosat_homogenization


# MAIN PROGRAM
def main():
  # command line arguments
  year0,year1 = 1780,2020
  ifile = []
  apply_norms = True
  tor = 0.1

  for arg in sys.argv[1:]:
    if arg.split("=")[0] == "-years":    # year calc
      year0,year1 = [int(x) for x in arg.split("=")[1].split(",")]
    if arg.split("=")[0] == "-i":        # input file
      ifile.append( arg.split("=")[1] )
    if arg.split("=")[0] == "-raw": # input file
      apply_norms = False

  # years and months
  years  = numpy.arange(year0,year1+1)
  months = numpy.arange(1,13)

  # read station data
  dsrcs = []
  for i in range(len(ifile)):
    dsrc = pandas.read_pickle( ifile[i], compression='bz2' )
    dsrc["xflag"] = i
    dsrcs.append(dsrc)
    print(dsrc.describe)

  dflt = pandas.concat(dsrcs)
  print(dflt.describe)

  # extract 1 row per station
  dsub = dflt.drop_duplicates( subset="stationcode" )
  print(dsub.columns)
  print(dsub.describe())

  # extract column info
  codes = dsub.loc[:,"stationcode"].values
  names = dsub.loc[:,"stationname"].values
  ctrys = dsub.loc[:,"stationcountry"].values
  flags = dsub.loc[:,"xflag"].values
  lats  = dsub.loc[:,"stationlat"].values
  lons  = dsub.loc[:,"stationlon"].values
  dists = glosat_homogenization.prepare_dists( lats, lons )

  # and full data
  dcodes = dflt.loc[:,"stationcode"].values
  dyears = dflt.loc[:,"year"].values
  dtemps = dflt.loc[:,["1","2","3","4","5","6","7","8","9","10","11","12"]].values
  if apply_norms:
    dnorms = dflt.loc[:,["n1","n2","n3","n4","n5","n6","n7","n8","n9","n10","n11","n12"]].values
    dtemps -= dnorms

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

  # now make filtered tables
  data0 = data.copy()
  data1 = data.copy()
  data0[:,flags!=0] = numpy.nan
  data1[:,flags!=1] = numpy.nan
  
  # interpolate
  cov = numpy.exp( -dists/900.0 )
  fill0 = data0.copy()
  fill1 = data1.copy()
  fill0[numpy.isnan(fill0)] = 0.0
  fill1[numpy.isnan(fill1)] = 0.0
  for j in range(data.shape[0]):
    twgt = glosat_homogenization.interpolatew( data0[j,:], cov, tor )
    fill0[j,:] = numpy.dot( fill0[j,:], twgt )
    twgt = glosat_homogenization.interpolatew( data1[j,:], cov, tor )
    fill1[j,:] = numpy.dot( fill1[j,:], twgt )

  # test
  for s in range(data.shape[1]):
    if numpy.count_nonzero(~numpy.isnan(data0[:,s])) > numpy.count_nonzero(~numpy.isnan(data1[:,s])):
      print(s,numpy.count_nonzero(~numpy.isnan(data0[:,s])),numpy.nanvar(data0[:,s]-fill1[:,s]))
    else:
      print(s,numpy.count_nonzero(~numpy.isnan(data1[:,s])),numpy.nanvar(data1[:,s]-fill0[:,s]))

# Main program launcher
if __name__ == '__main__':
    main()


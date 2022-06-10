"""
This module provides functions relating to homogenization and the
calculation of weather station normals. It was developed as part of the
GloSAT project.
"""
import numpy


# prepare distance matrix from lat/lon in degrees
def prepare_dists(lats, lons):
  """
  Prepare distance matrix from vectors of lat/lon in degrees assuming
  spherical earth
  
  Parameters:
    lats (vector of float): latitudes
    lons (vector of float): latitudes
  
  Returns:
    (matrix of float): distance matrix in km
  """
  las = numpy.radians(lats)
  lns = numpy.radians(lons)
  dists = numpy.zeros([las.size,las.size])
  for i in range(lns.size):
    dists[i,:] = 6371.0*numpy.arccos( numpy.minimum( (numpy.sin(las[i])*numpy.sin(las) + numpy.cos(las[i])*numpy.cos(las)*numpy.cos(lns[i]-lns) ), 1.0 ) )
  return dists


# return weights for a list of locations using ordinary krigging
def interpolatew( obs, cov, tor=0.0 ):
  """
  Interpolate values at locations described by the given covariance matrix
  using ordinary kriging with errors.
  
  Parameters:
    obs (vector of float): observations (some of which may be missing)
    cov (matrix of float): covariances or correlation matrix
    tor (float): error parameter, see https://hal.archives-ouvertes.fr/cel-02285439v2/document
  
  Returns:
    (vector of float): weights
  """
  # set up matrices
  data = obs.flatten()
  unobsflag = numpy.isnan(data)
  obsflag = numpy.logical_not( unobsflag )
  a = cov[obsflag,:][:,obsflag] + (tor**2)*numpy.identity(numpy.count_nonzero(obsflag))
  b = cov[obsflag,:]
  a = numpy.vstack( [
        numpy.hstack( [ a, numpy.ones([a.shape[0],1]) ] ),
        numpy.hstack( [ numpy.ones([1,a.shape[1]]), numpy.zeros([1,1]) ] )
    ] )
  b = numpy.vstack( [ b,numpy.ones([1,b.shape[1]]) ] )
  # solve for basis function weigths
  try:
    x = numpy.linalg.solve( a, b )
  except:
    x = numpy.dot( numpy.linalg.pinv(a), b )
  # calculate weights and store
  result = numpy.zeros(data.shape+data.shape)
  result[obsflag,:] = x[:-1,:]
  return result.reshape(obs.shape+obs.shape)


# remove monthly mean from each station month
def simple_norms( obs ):
  """
  Calculate simple station norms by calculating the mean for each month of non-missing data
  
  Parameters:
    obs[nmon,nstn] (matrix of float): observations (some of which may be missing)
  
  Returns:
    [nmon,nstn] (matrix of float)
  """
  # set initial baselines
  norms = numpy.full( obs.shape, numpy.nan )
  mnths = numpy.arange(obs.shape[0])%12
  # set baselines
  for s in range(obs.shape[1]):
    for m in range(12):
      mmask = (mnths==m)
      norms[mmask,s] = numpy.nanmean(obs[mmask,s])
  return norms


# fit station fragment norms
def fit_norms( obs, flags, nfourier=0 ):
  """
  Fit station fragment norms station fragment using mean and seasonal cycle 
  
  Parameters:
    [nmon,nstn] obs (vector of float): Station temperature series
    [nmon,nstn] flags (vector of uint8): Station fragment flags
    nfourier (int): Number of fourier orders used to fit annual cycle changes

  Returns:
    [nmon,nstn] (vector of float): vector of norms by month
  """
  norms = numpy.zeros_like( obs )
  for s in range(obs.shape[1]):
    y = obs[:,s]
    flagstn = flags[:,s]
    frags = numpy.unique(flagstn)
    nfrags = frags.shape[0]
    npar = 12 + (nfrags-1)*(1+2*nfourier)
    x = numpy.zeros([y.shape[0],npar])
    # annual cycle
    p = 0
    for m in range(12):
      x[m::12,p] = 1.0
      p += 1
    # constant shifts for fragments
    for f in range(nfrags-1):
      x[:,p] = 1.0*(flagstn==f)
      p += 1
    # cosine shifts for fragments
    for n in range(nfourier):
      for f in range(nfrags-1):
        x[:,p  ] = _cost[:,n]*(flagstn==f)
        x[:,p+1] = _sint[:,n]*(flagstn==f)
        p += 2
    xm = x[ ~numpy.isnan(y), : ]
    ym = y[ ~numpy.isnan(y) ]
    if ym.shape[0] < xm.shape[1]: return numpy.zeros_like(y)
    p = numpy.linalg.lstsq(xm,ym,rcond=None)[0]
    norms[:,s] = numpy.dot(x,p)
  return norms


# solve for station fragment norms
def solve_norms( obs, flags, cov, tor, nfourier, diagnostics=1 ):
  """
  Solve for station fragment norms using Kriging weights and full matrix
  least squares.
  
  Parameters:
    obs[nmon,nstn] (matrix of float): observations (some of which may be missing)
    flags[nmon,nstn] (matrix of int): flags demarkating station fragments 0...n
    cov (matrix of float): covariances or correlation matrix
    tor (float): error parameter, see https://hal.archives-ouvertes.fr/cel-02285439v2/document
    nfourier (int): number of Fourier orders to use in norms
  
  Returns:
    tuple of:
      [nmon,nstn] norms (matrix of float)
      [nmon,nstn] uncertainties in norms (matrix of float)
      [npar] parameter mappings (vector of 2-tuples)
      [npar] parameter values (vector)
      [npar,npar] covariance matrix (matrix)
  """
  nmon, nstn = obs.shape

  # calculate list of equations - one per observation
  #              and parameters - one per station fragment
  eqns = []
  pars = []
  for s in range(nstn):
    # add one equation per observation
    for m in range(nmon):
      if not numpy.isnan( obs[m,s] ):
        eqns.append( (m,s) )
    # add one parameter per fragment
    frags = numpy.unique( flags[:,s] )
    for f in frags:
      pars.append( (f,s) )
  eqns = numpy.array(eqns,dtype=int)
  pars = numpy.array(pars,dtype=int)
  neqn = eqns.shape[0]
  npar = pars.shape[0]
  print(eqns)
  print(pars)
  print(eqns.shape)
  print(pars.shape)

  # make the least squares coefficients
  A = numpy.zeros( [neqn,npar] )
  B = numpy.full( [neqn], numpy.nan )

  # construct the matrices
  mons = numpy.arange(nmon)
  for m in range(nmon):
    # get kriging weights for this month
    wijt = interpolatew( obs[m,:], cov, 0.1 )
    # the data premultiply the weights, so each column of w is a set of weights
    wijt -= numpy.identity( wijt.shape[0] )
    # rhs terms
    bs = obs[m,:].copy()
    bs[numpy.isnan(bs)] = 1.0e30  # corresponding w should be zero
    bs = numpy.dot( bs, wijt )
    # fill in coefficient matrix and rhs for equations involving this month
    mmsk = (eqns[:,0] == m)
    for e in numpy.nonzero(mmsk)[0]:
      s2 = eqns[e,1]
      B[e] = bs[s2]
      pmsk = flags[m,pars[:,1]] == pars[:,0]
      A[e,pmsk] = wijt[pars[pmsk,1],s2]
    """
    # conceptually, the previous code does the following:
    for e in range(neqn):
      m2,s2 = eqns[e]
      if m2 == m:
        B[e] = bs[s2]
        for p in range(npar):
          f1,s1 = pars[p]
          if flags[m,s1] == f1:
            A[e,p] = wijt[s1,s2]
    """

  # now augment the matrices for estimation of Fourier coefficients for annual cycle
  nblock = 2*nfourier+1
  dt = (numpy.arange(nmon)+0.5)/12.0
  wmfourier = numpy.ones( [nmon,nblock] )
  for f in range(nfourier):
    wmfourier[:,2*f+1] = numpy.cos(2*numpy.pi*(((f+1)*dt)%1.0))
    wmfourier[:,2*f+2] = numpy.sin(2*numpy.pi*(((f+1)*dt)%1.0))
  wefourier = wmfourier[eqns[:,0],:]
  A = A[:,:,numpy.newaxis] * wefourier[:,numpy.newaxis,:]
  # make extra constraint rows
  Aadd = numpy.zeros( [nblock,npar,nblock] )
  Badd = numpy.zeros( [nblock] )
  for i in range(nblock): Aadd[i,:,i] = 1.0
  # flatten new parameters
  A.shape = [neqn,nblock*npar]
  Aadd.shape = [nblock,nblock*npar]

  # add final equations constraining sum of offsets to zero
  #   Aadd = numpy.ones( [1,npar] ) # for no fourier coeffs
  #   Badd = numpy.zeros( [1] )     # for no fourier coeffs
  print(A.shape, B.shape)
  A = numpy.vstack( [ A, Aadd ] )
  B = numpy.hstack( [ B, Badd ] )
  print(A.shape, B.shape)

  # solve the equations
  print("SOLVE",numpy.count_nonzero(numpy.isnan(A)),numpy.count_nonzero(numpy.isnan(B)))
  # now solve
  Q = numpy.linalg.pinv( numpy.dot( A.T, A ) )
  X = numpy.dot( Q, numpy.dot( A.T, B ) )

  # get uncertainties
  rss = numpy.sum( numpy.power( numpy.dot(A,X)-B, 2 ) ) / ( neqn-npar )
  Q *= rss
  E = numpy.sqrt( numpy.diagonal( Q ) )

  # reshape to match data
  X.shape = [npar,nblock]
  E.shape = [npar,nblock]
  for b in range(nblock): print(X[:,b])
  for b in range(nblock): print(E[:,b])

  # store norms and uncertainties
  norms = numpy.full( [nmon,nstn], numpy.nan )
  norme = numpy.full( [nmon,nstn], numpy.nan )
  for p in range(npar):
    f1,s1 = pars[p]
    mmsk = flags[:,s1]==f1
    norms[mmsk,s1] = numpy.dot(wmfourier,X[p,:])[mmsk]
    norme[mmsk,s1] = numpy.dot(wmfourier,E[p,:])[mmsk]

  # and return them
  if diagnostics: return ( norms, norme, pars, X, Q )
  return ( norms, norme )


# solve for station fragment norms
def solve_norms_iter( obs, flags, cov, tor, nfourier, niter=10 ):
  """
  Solve for station fragment norms using Kriging weights and
  iteration with local expectation.
  
  Parameters:
    obs[nmon,nstn] (matrix of float): observations (some of which may be missing)
    flags[nmon,nstn] (matrix of int): flags demarkating station fragments 0...n
    cov (matrix of float): covariances or correlation matrix
    tor (float): error parameter, see https://hal.archives-ouvertes.fr/cel-02285439v2/document
    nfourier (int): number of Fourier orders to use in norms
    niter (int): number of iterations to perform
  
  Returns:
    tuple of:
      [nmon,nstn] norms (matrix of float)
      [nmon,nstn] uncertainties in norms (matrix of float, EMPTY)
  """
  norms,norme = numpy.full( obs.shape, 0.0 ), numpy.full( obs.shape, numpy.nan )
  dfull = obs - norms

  # calculate local expectations
  # ----------------------------
  # Now we calculate a local expectation at the location of each station using the
  # anomalies.
  dlexp,var = local_expectation( dfull, cov, tor )

  # Iteratively find breakpoints
  # ----------------------------
  # We loop over n cycles, finding breakpoints from the difference between a station and
  # its expectation, then updateing the norms and expectations.
  for cycle in range(niter):
    norms = fit_norms( obs - dlexp, flags, nfourier=nfourier )
    dfull = obs - norms
    dlexp,var = local_expectation( dfull, cov, tor )

  # and return them
  return ( norms, norme )


# solve for station fragment norms
def solve_norms_iter_err( obs, flags, cov, tor, nfourier, niter=10, nerr=6 ):
  """
  Solve for station fragment norms using Kriging weights and
  iteration with local expectation.
  
  Parameters:
    obs[nmon,nstn] (matrix of float): observations (some of which may be missing)
    flags[nmon,nstn] (matrix of int): flags demarkating station fragments 0...n
    cov (matrix of float): covariances or correlation matrix
    tor (float): error parameter, see https://hal.archives-ouvertes.fr/cel-02285439v2/document
    nfourier (int): number of Fourier orders to use in norms
    niter (int): number of iterations to perform. DEFAULT=10
    nerr  (int): number of cycles to estimate errors. DEFAULT=5
  
  Returns:
    tuple of:
      [nmon,nstn] norms (matrix of float)
      [nmon,nstn] uncertainties in norms (matrix of float)
  """
  nmon, nstn = obs.shape

  # calculate initial norms
  norms,*others = solve_norms_iter( obs, flags, cov, tor, nfourier, niter )

  nosds = numpy.full( obs.shape, numpy.nan )
  for s in range(nstn):
    frags = numpy.unique( flags[:,s] )
    for f in frags:
      msk = flags[:,s]==f
      nosds[msk,s] = numpy.nanstd( obs[msk,s]-norms[msk,s] )

  normn = []
  for c in range(nerr):
    sim = numpy.random.normal(norms,nosds)
    sim[numpy.isnan(obs)] = numpy.nan
    normx,*others = solve_norms_iter( sim, flags, cov, tor, nfourier, niter )
    normn.append( normx )

  # remove cycle in norm uncertainties (also incresing sample size)
  norme = numpy.nanstd(normn,axis=0)
  for s in range(nstn):
    frags = numpy.unique( flags[:,s] )
    for f in frags:
      msk = flags[:,s]==f
      norme[msk,s] = numpy.mean(norme[msk,s])

  """
  import matplotlib.pyplot as plt
  for s in range(nstn):
    t = norms[:,s]
    a = numpy.empty_like(t)
    for m in range(12): a[m::12] = t[m::12] - numpy.nanmean(t[m::12])
    plt.plot( a[:], 'k-', lw=3 )
    plt.plot( obs[:,s], 'bx', ms=3 )
    plt.plot( sim[:,s], 'ro', ms=3 )
    plt.fill_between( numpy.arange(nmon), a[:]-norme[:,s], a[:]+norme[:,s], color='k', alpha=0.2 )
    plt.savefig( "sim{:04d}.png".format(s) )
    plt.close()
  """

  # and return them
  return ( norms, norme )



# calculate local expectation using kriging with approximate hold-out
def local_expectation( obs, cov, tor ):
  """
  Calculate local expectation using approximate holdout kriging.
  
  Parameters:
    obs[nmon,nstn] (matrix of float): observations (some of which may be missing)
    cov (matrix of float): covariances or correlation matrix
    tor (float): error parameter, see https://hal.archives-ouvertes.fr/cel-02285439v2/document
  
  Returns:
    [nstn],[nstn] (vector of float): 2 tuple containing estimates, variances
  """
  # infill from updated station anomalies
  fill = numpy.full( obs.shape, numpy.nan )
  var  = numpy.full( obs.shape, numpy.nan )
  for j in range(obs.shape[0]):
    twgt = interpolatew( obs[j,:], cov, tor )  # get kriging weights
    numpy.fill_diagonal( twgt, 0.0 )           # zero self weights
    twgt = twgt / numpy.sum( twgt, axis=0 )    # renormalize
    tobs = obs[j,:].copy()
    tobs[numpy.isnan(tobs)] = 0.0
    fill[j,:] = numpy.dot( tobs, twgt )
    var[j,:]  = numpy.diagonal( cov ) - numpy.diagonal( numpy.dot( cov, twgt ) )
  return fill, var

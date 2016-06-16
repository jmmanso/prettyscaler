import pickle
import numpy as np
from scipy import interpolate





class PrettyScaler:

	''' 
	This class offers data preprocessing methods that 
	build a transformation function from the Reals domain 
	to (-1,1). Any 1-dimensional data set can be fitted, 
	producing a unique and bijective transformation. 

	Includes fit/transform methods following Sciki-learn's practices,
	as well as parameter I/O through save_pickle/load_pickle.

	'''

	def fit(self, 
		data,
		wing_percentile=0.01,
		ftol=1e-5,
		sample_size=1e3
		):
		''' 
		DESCRIPTION
			This function takes a data object and fits 2 transformations
			that map (-Infitnity,Infinity) to (-1,1). Given the local density 
			of the data in the Real domain, the final transformation efficiently 
			uses the (-1,1) domain to squeeze in the new density. The output 
			transformations produce samples with flat and bulge-like distribution
			functions over (-1,1).

		ARGUMENTS
			data (Pandas data frame, Pandas series, numpy array, tuple, list):
				data object containing a 1-D array of points


		KEYWORD ARGUMENTS 

			wing_percentile (float): p-value of the wings, which determines their boundary

			ftol (float): minimum difference to be enforced between consecutive values in the 
				cumulative distribution function of the data. Helps with numerical precision. 

			sample_size (int): size of grid to pin and evaluate the densities in original
				and final spaces. A random sample of this size will be drawn from the input 
				data to define the grid. This is a cheap and dirty way of sampling the data 
				density.



		RETURNS
			None. Adds fitted parameters to the class object. 
		'''
		# <TECH DEBT>
		# In the future, we will make the output domain bounds
		# an input argument, and thus we will not hard-code in 
		# its values all throughout the code. 
		bounds = [-1,1]
		# </TECH DEBT>
		# Ensure data has the right format
		self.data = self.format(data)

		# use a random sample of the data
		# to define the the grid E Reals
		np.random.shuffle(self.data)
		self.xgrid = self.data[:sample_size]

		# ensure all grid points are unique
		self.xgrid = np.unique(self.xgrid)
		# compute the cumulative sum of the data for every xgrid value. It will
		# be bound within 0,1
		self.y_cumsum = self.accumulate(self.xgrid, self.data)
		# 
		# stretch, making sure that all consecutive elements 
		# are separated by more than ftol
		self.y_stretched = self.stretch(self.y_cumsum,ftol)
		# normalize, since the stretched's max value will have exceeded 1 by a litle bit
		self.y_normalized = self.normalize(self.y_stretched)
		# scale within bounds
		self.y_scaled = self.scale(self.y_normalized,bounds)
		#
		# define boundaries
		self.compute_boundaries_and_params(wing_percentile)
		# build full parameter dictionary
		self.build_params()
		#
		# set up functions
		self.build_functions()
		

	def build_params(self):
		''' Adds parameter objects to class dictionary '''
		self.all_params = {'trunk_x':list(self.trunk_x),'trunk_y':list(self.trunk_y),
		'wing_offset_right':self.wing_offset_right,'wing_coeff_right':self.wing_coeff_right,
		'wing_offset_left':self.wing_offset_left,'wing_coeff_left':self.wing_coeff_left,
		'rightwing_boundary_value':self.rightwing_boundary_value,
		'leftwing_boundary_value':self.leftwing_boundary_value}



	def compute_boundaries_and_params(self, wing_percentile):
		''' Determines the domain boundaries for the left wing, right wing and 
		trunk functions, in addition to other parameters. All this depends on the 
		wing percentile argument, which represents the p-value contained within 
		each wing function. '''
		# get the xgrid index that defines the wings based on the percentile setting
		self.leftwing_boundary_idx = self.value2index(self.y_normalized, wing_percentile)
		self.rightwing_boundary_idx = self.value2index(self.y_normalized, 1.0 - wing_percentile)
		# get the associated x value
		self.leftwing_boundary_value = self.xgrid[self.leftwing_boundary_idx]
		self.rightwing_boundary_value = self.xgrid[self.rightwing_boundary_idx]
		# and the y value, which ideally would be the same as the wing percentile, but 
		# may differ due to discretization of space 
		self.leftwing_y_value =  self.y_scaled[self.leftwing_boundary_idx]
		self.rightwing_y_value =  self.y_scaled[self.rightwing_boundary_idx]
		# define the x,y grids for the trunk (main body within wings)
		self.trunk_x = self.xgrid[self.leftwing_boundary_idx:self.rightwing_boundary_idx+1]
		self.trunk_y = self.y_scaled[self.leftwing_boundary_idx:self.rightwing_boundary_idx+1]
		#
		# get the gradient from the last 2 points
		self.dx_right = self.xgrid[self.rightwing_boundary_idx+1]-self.xgrid[self.rightwing_boundary_idx]
		self.dy_right = self.y_scaled[self.rightwing_boundary_idx+1]-self.y_scaled[self.rightwing_boundary_idx]
		# the true derivative should be positive, and we flip it here to negative to resemble the basic 
		# exponential case Exp[-z].
		self.dydx_right = - self.dy_right/self.dx_right
		# Flip the y values to resemble the trend in Exp[-z]
		self.y_right = 1.0 - self.rightwing_y_value
		# Calculate coefficient and offset in the generalized exponential 
		# to match the value and gradient of the trunk function at the boundary:
		self.wing_offset_right = self.rightwing_boundary_value - np.log(self.y_right)*self.y_right/self.dydx_right
		self.wing_coeff_right = - self.dydx_right/self.y_right
		#
		#
		# Do the same thing with the left wing. Calculate gradient:
		self.dx_left = self.xgrid[self.leftwing_boundary_idx+1]-self.xgrid[self.leftwing_boundary_idx]
		self.dy_left = self.y_scaled[self.leftwing_boundary_idx+1]-self.y_scaled[self.leftwing_boundary_idx]
		# again, make the derivative negative
		self.dydx_left = - self.dy_left/self.dx_left
		self.y_left = 1.0 + self.leftwing_y_value
		self.wing_offset_left = self.leftwing_boundary_value - np.log(self.y_left)*self.y_left/self.dydx_left
		self.wing_coeff_left = - self.dydx_left/self.y_left

	def build_functions(self):
		''' Uses the fitted or loaded parameters and constructs
		the transformation functions: left wing, right wing and trunk 
		functions. Then, these functions are glued together to produce 
		the final full transformation. '''

		# Build function for the main domain body
		self.trunk_function = interpolate.interp1d(self.all_params['trunk_x'],self.all_params['trunk_y'],
			kind='linear', bounds_error=False, fill_value=0.0)
		# Build function for the left wing
		self.left_wing_function = lambda x: \
		self.generalized_exponential(-x,self.all_params['wing_coeff_left'],\
			self.all_params['wing_offset_left']-2*self.all_params['leftwing_boundary_value']) - 1.0
		# Build function for the right wing
		self.right_wing_function = lambda x: \
		1.0 - self.generalized_exponential(x,self.all_params['wing_coeff_right'],\
			self.all_params['wing_offset_right'])		

		# Make final full-domain function as a composite of left, trunk and right functions
		self.transfer_function = lambda x: self.left_wing_function(x) \
		if x<=self.all_params['leftwing_boundary_value'] else self.right_wing_function(x) \
		if x>=self.all_params['rightwing_boundary_value'] else 1.0*self.trunk_function(x)
		#
		# fit the bulge transformation
		self.fit_bulge()


	def fit_bulge(self):
		''' Map the flat space onto a bulge-like one.

		Essentially, we want the probability density distro of the data 
		to be PDF[x]:=1-Cos[Pi*(x+1)], for x E {-1,1}.

		We want a transformation that maps a flat space [-1,1] to PDF[x].
		The mapping is given by the function iCDF[x], which is the inverse
		of CDF[x]: 
		CDF[x]:=Integrate[PDF[z],z] = x + Sin[Pi*x]/Pi within [-1,1]

		Since CDF[x] is not analytically invertible, we will map the flat
		[-1,1] space onto CDF[x], and then define our transformation as an
		interpolation object of the swapped relation.

		 '''
		# This would be CDF[x]:
		self.bulge_auxfunc = lambda x: x + np.sin(x*np.pi)/(np.pi)
		# Map onto array from flat [-1,1] distro
		self.bulge_y = np.array([self.bulge_auxfunc(u) for u in self.y_scaled])
		# Build interpolation object
		self.bulge_intp = interpolate.interp1d(self.bulge_y, self.y_scaled,kind='linear', \
			bounds_error=False, fill_value=0.0)


	def transform(self, data, kind='flat'):
		''' Takes 1-D data set and transforms according 
		to the fitted functions '''

		# Check that the parameter dictionary is in memory
		if not hasattr(self, 'all_params'):
			raise RuntimeError('You need to fit some data or load parameters before attempting to use this method.')

		# Check that you're not passing some weird name
		if kind not in ['flat','bulge']:
			raise TypeError('Keyword kind should be flat or bulge')

		# Ensure data has right format
		data = self.format(data)
		# Do the flat transformation anyway
		transformed_data = np.array([1.0*self.transfer_function(u) for u in data])
		if kind=='bulge':
			# Execute bulge transformation on top of flat
			print 'Mapping to bulge distribution'
			transformed_data = np.array([1.0*self.bulge_intp(u) for u in transformed_data])

		return transformed_data


	def format(self, data):
		''' Transforms a one-dimensional structure
		into a numpy array. Inputs can be numpy array,
		list, tuple, dataframe, series. '''
		obj_type = type(data).__name__
		if obj_type in ['DataFrame','Series']:
			data = data.values
		elif obj_type == 'ndarray':
			data = data 
		elif obj_type in ['list','tuple']:
			data = np.array(data)
		else:
			raise IOError('Input data not recognized. Should be ndarray, list, tuple, Series or DataFrame.')
		#
		# ensure data is one-dimensional array
		if len(data.shape)>1:
			print 'Warning: value array is not one-dimensional and will be flattened.'
			data = data.ravel()
		#
		# check that you have a 1-D array of numbers by trying a vector operation
		try:
			dot_product = np.dot(data, np.ones(len(data)))
		except:
			raise ValueError('Data values could not be converted to numerical 1-D array.')
		#
		# make sure that you have at least a few data points:
		if len(data)<20:
			raise RuntimeError('Data set is too small.')

		return 1.0*data


	def save_pickle(self, file_path):
		# save as pickle
		with open(file_path,'wb') as f:
			pickle.dump(self.all_params,f)

	def load_pickle(self, file_path):
		# Load params
		with open(file_path,'rb') as f:
			self.all_params = pickle.load(f)
		# Build functions
		self.build_functions()


	@staticmethod
	def accumulate(x_array,data):
		''' Returns the cumulative sum of points
		in data (array of size(M,)) for every 
		point in x_array (array of size(N,))'''
		return np.array([data[data<=k].size*1.0 for k in x_array])/len(data)

	@staticmethod
	def stretch(y_array,ftol):
		''' Returns a modified copy of y_array
		where data points have been offset enough
		to sequentially differ by more than ftol  '''
		# determine the deltas (sequential absolute differences)
		# of the cumulative array
		deltas = y_array[1:] - y_array[:-1]
		# identifiy those indices where the delta is less than the predefined
		# precision tolerance
		idxs = np.where(deltas<ftol)[0]
		# we will increase the deltas in those indices, creating 
		# a final "stretched-out" array
		for idx in idxs:
			# the +1 represents the indexing offset between cumsum and the 
			# array of deltas
		    y_array[idx+1:] = y_array[idx+1:] + ftol
		#
		return y_array

	@staticmethod
	def normalize(y_array):
		''' Scales data to have a max value of 1'''
		return y_array/max(y_array)


	@staticmethod
	def scale(y_array,bounds=[-1,1]):
		''' Rescales data to new finite bounds '''
		lower_bound, upper_bound = bounds
		bounds_range = upper_bound - lower_bound
		bounds_midpoint = (upper_bound - lower_bound)/2.0
		y_array = y_array * bounds_range - bounds_midpoint
		return y_array

	@staticmethod
	def value2index(input_array, value):
		'''Returns the index of an array whose element value is closest to \"value\"'''
		# Handle a couple of edge-cases:
		if value < min(input_array):
			return 0 
		#
		if value > max(input_array):
			return input_array.size-1
		#
		#
		stat = abs(input_array - value)
		return np.where( stat == min(stat) )[0][0]

	@staticmethod
	def generalized_exponential(x,coeff,offset):
		return np.exp(-(x-offset)*coeff)













import signal
import sys

import numpy as np
from typing import List

from time import sleep

# from matplotlib import pyplot
from matplotlib.patches import Rectangle

class MapType:
	UNKNOWN = -1
	UNOCCUPIED = 0
	OCCUPIED = 1

class QuadMap:
	"""
	This class maintains and updates a QuadTree representation of occupied and unoccupied space.
	It starts with all space being represented as unknown and updates the values of points in space
	based upon point_updates and ray_updates. 

	To implement this class students are provided with a framework for a QuadMapNode class. Using the
	provided QuadMapNode class is not necessary, students may create their own datastructure if they
	prefer. But the overall behavior of the QuadMap itself must be correct. 

	"""
	def __init__(self, 
				max_depth = 3,
				size = 4.0,
				origin = np.array([0.0,0.0])):
		"""
		Constructor for the QuadMap. All space is unknown when it is first initialized.

		The size of the quadmap represents its height and width.
		The origin represents the centerpoint of the quadmap.
		The default values should give a QuadMap that extends from [-5,5] at the bottom left 
		corner to [5,5] at the upper right corner. With a maximum resolution of size / 2^max_depth. 

		The constructor is graded on functionality, not a specific approach.

		Worth 10 pts

		Input
		  :param max_depth: The maximum depth of the quadtree.
		  :param size: The total height and width of the quadmap (i.e. the size of the root node)
		  :param origin: The centerpoint for the quadmap 
		"""
		self.root = QuadMapNode(None, origin, size)
		self.size = size
		self.depth = 0
		self.max_depth = max_depth
		# Ray step size is relative to max depth because of computing resources
		self.ray_step_size = self.size / np.power(2, self.max_depth+1)
		self.figure = None

	def point_update(self, point: np.ndarray, state: int):
		"""
		Updates the value of the given point in the QuadMap. Update the value at the maximum depth.

		If the node located at this point already has this state do nothing.

		If the node located at this point is a leaf node at the maximum depth then update is value.

		If the node located at this point is a leaf node not at maximum depth then split it and repeat. 

		After updating the value of the point with its new state the quadmap should collapse any 
		nodes where all the children have the same value. 

		Worth 50 pts

		Input
		  :param point: A 2 element np.ndarray containing the x and y coordinates of the point
		  :param state: An integer indicating the MapType of the point
		"""
		current_node = self.root
	
		for depth in range(self.max_depth):
			if current_node.state == state:
				return
			if not current_node.children:
				current_node.split()
		
			# Check bounds - exit early if point is outside node
			delta = abs(point - current_node.origin)
			if delta[0] > current_node.size / 2 or delta[1] > current_node.size / 2:
				return
		
			# Calculate quadrant offset
			quadrant = 0
			if point[0] < current_node.origin[0]:
				quadrant += 2
			if point[1] < current_node.origin[1]:
				quadrant += 1
			
			current_node = current_node.children[quadrant]
	
		# Update the leaf node and collapse if needed
		current_node.state = state
		current_node.combine()

	def ray_update(self, origin: np.ndarray, endpoint: np.ndarray):
		"""
		This function update the map based upon the return of a rangefinding beam.

		This function should update the map to indicate that all the space between the origin
		and the endpoint is unoccupied. Then update the endpoint as occupied.

		Worth 10 pts

		Input
		  :param origin: A 2 element ndarray indicating the location of the robot
		  :param endpoint: A 2 element ndarray indicating the location of the end of the beam
		"""
		ray_vector = endpoint - origin
		ray_distance = np.linalg.norm(ray_vector)
		step_count = int(ray_distance / self.ray_step_size)
		
		# Mark all intermediate points as unoccupied
		for step in range(step_count):
			sample_point = origin + (step * self.ray_step_size / ray_distance) * ray_vector
			self.point_update(sample_point, MapType.UNOCCUPIED)
		
		# Mark endpoint as occupied
		self.point_update(endpoint, MapType.OCCUPIED)

	def get_state(self, point: np.ndarray):
		"""
		Returns the MapType state (-1 for unknown, 0 for unoccupied, 1 for occupied) for a given point

		Worth 10 pts

		Input
		  :param agent: An instance of the Agent class. The agent that will execute the velocity command
		  :param obstacle: A single Obstacle object that will be used to compute the velocity obstacle

		Output
		  :return: state: an integer representing the state of that point in space
		"""
		current = self.root
		while current.children:
			quadrant_idx = 0
			if point[0] < current.origin[0]:
				quadrant_idx += 2
			if point[1] < current.origin[1]:
				quadrant_idx += 1
			current = current.children[quadrant_idx]
		return current.state

	def to_occupancygrid(self):
		"""
		Converts the QuadMap into the data element of an occupancy grid. 
		A flattened representation used by ROS. 

		This is a 1-dimensional array with a length equal to the maximum number of nodes in the
		QuadMap. 

		Follow the guidance for the occupancy grid message
		https://github.com/ros2/common_interfaces/blob/master/nav_msgs/msg/OccupancyGrid.msg

		Worth 10 pts

		Output
		  :return: data: a list of integers representing the node values
		"""
		num_leaf = np.power(2, self.max_depth)
		grid = np.zeros((num_leaf,num_leaf))

		# Each stack item will be a tuple of (node, corresponding grid slice)
		stack = [(self.root, grid)]

		while stack:
			current_node, current_grid = stack.pop()
		
			# If leaf node, fill the grid with the state value
			if current_node.children is None or len(current_node.children) == 0:
				current_grid[:] = current_node.state
			else:
				# Calculate half point for quadrant division
				half = int(len(current_grid) / 2)
			
				# Push children to stack with their corresponding grid slices
				# We push in reverse order to maintain the same processing order
				# as the recursive version (since stack is LIFO)
				stack.append((current_node.children[3], current_grid[half:,:half]))
				stack.append((current_node.children[2], current_grid[:half,:half]))
				stack.append((current_node.children[1], current_grid[half:,half:]))
				stack.append((current_node.children[0], current_grid[:half,half:]))

		return grid

	# def plot(self):
	# 	"""
	# 	A plot helper function. Much of the code is already in place but expects a working plot function for
	# 	the QuadMapNode class. This function can be modified as necessary to work with your implementation.

	# 	Worth 10 pts
	# 	"""
	# 	# Initialize figure and axes if not already created
	# 	if self.figure is None:
	# 		self.figure = pyplot.figure(figsize=(15, 5))
	# 		self.axis = [
	# 			self.figure.add_subplot(1, 2, 1),  # Left subplot (row=1, col=2, index=1)
	# 			self.figure.add_subplot(1, 2, 2)   # Right subplot (row=1, col=2, index=2)
	# 		]
		
	# 	grid = self.to_occupancygrid()
	# 	pyplot.xlim([0, 10])
	# 	pyplot.ylim([0, 10])
	# 	self.root.plot_node(self.axis[1])
	# 	self.axis[0].imshow(grid)
	# 	pyplot.draw()
	# 	pyplot.pause(0.0001)

class QuadMapNode:
	"""
	A partially implemented Node class for constructing a quadtree. 
	The student is welcome to modify this template implementation by adding
	new class variables or functions, or modifying existing variables and functions.

	The individual functions in QuadMapNode are not graded.

	Each QuadMapNode represents a node in a tree. Where every node other than the root
	has one parent and either 0 children or 4 children. Each node is half the length and width of
	its parent (1/4 the area). 
	"""
	def __init__(self, parent, origin: np.ndarray, size: float = None):
		"""
		Default constructor for the QuadMapNode.

		Input
		  :param parent: The parent node (if this node is root parent=None)
		  :param origin: A 2 element np.ndarray containing the centerpoint of the node
		  :param size: A float describing the height and width of the node

		"""
		if parent is None:
			self.parent = None
			assert size is not None
			self.size = size
		else:
			self.parent = parent
			self.size = parent.size / 2.0 

		self.origin = origin # The origin is the centerpoint of the node
		self.children = [] # A list of all child nodes. Should only contain 0 or 4 children.
		self.state = MapType.UNKNOWN # A new node will not have a type assigned initially


	def split(self):
		"""
		Splits  node into 4 smaller nodes. This function should only be called if the current node is a leaf.

		A useful function to implement for your quadmap
		"""
		quarter_size = self.size / 4
		half_size = self.size / 2
		
		# Define quadrant offsets: [NE, SE, NW, SW]
		quadrant_offsets = [
			np.array([1.0, 1.0]),   # Northeast
			np.array([1.0, -1.0]),  # Southeast  
			np.array([-1.0, 1.0]),  # Northwest
			np.array([-1.0, -1.0])  # Southwest
		]
		
		# Create child nodes for each quadrant
		for offset in quadrant_offsets:
			child_origin = self.origin + quarter_size * offset
			self.children.append(QuadMapNode(self, child_origin, half_size))

	def combine(self):
		"""
		Checks to see if all the children of the node have the same type, if they do then delete them and assign
		the current node that type so that it becomes a leaf node.

		A useful function to implement for your quadmap
		"""
		# Safer to use iterative stack implementation instead of recursion. 
		# Prevents stack overflows for large map sizes or resource limited systems.
		stack = [self]

		while stack:
			current_node = stack.pop()
			# Handle case when node has children
			if len(current_node.children) > 0:
				child_state = current_node.children[0].state
				all_same = True

				for child in current_node.children:
					if child.state != child_state:
						all_same = False
						break

				if all_same:
					current_node.children = []
					current_node.state = child_state

					# Add parent to stack if it exists
					if current_node.parent:
						stack.append(current_node.parent)

			# If there were no children or after we've processed the children
			# We still need to try to combine the parent
			elif current_node.parent:
				stack.append(current_node.parent)

	def plot_node(self, ax):
		"""
		A helper function for visualizing the current state of the QuadMap. This function recursively calls itself
		against the children of the node. It may be necessary to modify this function to work with your implementation
		of the class.

		Note that only the leaf nodes are visualized!

		Input
		  :param ax: A matplotlib Axes object to plot the node
		"""
		if len(self.children) > 0:
			for child in self.children:
				child.plot_node(ax)
		else:
			if self.state == MapType.UNKNOWN:
				facecolor = 'grey'
			if self.state == MapType.UNOCCUPIED:
				facecolor = 'white'
			if self.state == MapType.OCCUPIED:
				facecolor = 'black'
			#print(self.origin,self.size,self.state)
			ax.add_patch(Rectangle((self.origin[0]-self.size/2,self.origin[1]-self.size/2),
								   self.size, self.size,
								   edgecolor = 'black',
								   facecolor = facecolor,
								   fill=True,
								   lw=5))

def main():
	
	map = QuadMap(max_depth=5, size=10.0, origin=np.array([5.0,5.0]))

	x = np.array([5,6.5])
	ray = np.array([2,3])
	for i in range(30):
		map.ray_update(x, x+ray)
		map.plot()
		x+=np.array([0.025,-0.025])
		ray = np.array([[np.cos(0.1),-np.sin(0.1)],[np.sin(0.1),np.cos(0.1)]])@ray

if __name__ == '__main__':
	main()

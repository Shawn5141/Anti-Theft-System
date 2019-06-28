class humanData:
	def __init__(self, x, y, idx, feature):
		self.x = x
		self.y = y
		self.id = idx
		self.updated = True
		self.missing = 0
		self.isSuspect = False
		self.itemList = []
		self.stolenitemDict={}
		self.firstFeature = feature
		self.feature = feature
	def update_position(self, nx, ny):
		self.x = nx
		self.y = ny


class itemData:
	def __init__(self, x, y, idx,name):
		self.x = x
		self.y = y
		self.id = idx
		self.updated = True
		self.missing = 0
		self.alarm_flag = False
		self.name=name
		self.owner=0
	def update_position(self, nx, ny):
		self.x = nx
		self.y = ny

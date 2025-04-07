class Territory:
    def __init__(self,name,pos):
        self.name = name            # Name of the territory, used for displaying ONLY (not dict key)
        self.adjecency_list = []    # List of all adjecent territories
        self.pos = pos              # Position on map, for displaying troops and attacks
        self.troops = 0             # Amount of troops on territory
        self.troops_to_add = 0      # Troops to add to territory       
        self.player_index = None          # Color of player that owns troops
    
    # Adds a connection to the adjecency list
    def connectto(self, territory_o: 'Territory'):
        self.adjecency_list.append(territory_o)

    # Checks if a territory is connected
    def isConnected(self, territory_o: 'Territory'):
        if territory_o in self.adjecency_list:
            return True
        return False
    
    def reset(self):
        self.troops = 0
        self.player_index = None

    def __repr__(self):
        return f"{self.name} ==> {self.player_index}: {self.troops}"
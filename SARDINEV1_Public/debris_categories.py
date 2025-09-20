"""
Space Debris Categories Module
Defines different types of orbital debris based on NASA/ESA classifications
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

__all__ = ['DebrisObject', 'DebrisCategory', 'SpacecraftDebris', 'create_debris_categories']

@dataclass
class DebrisObject:
    """Represents a single piece of space debris"""
    object_type: str
    altitude: float
    size_category: str
    mass: float
    orbital_period: float
    inclination: float
    
class DebrisCategory:
    """Base class for debris categories"""
    
    def __init__(self, name: str, color: str, description: str):
        self.name = name
        self.color = color
        self.description = description
        self.objects = []
        
    
    def add_object(self, obj: DebrisObject):
        """Add a debris object to this category"""
        self.objects.append(obj)
    
    def get_altitude_distribution(self) -> List[float]:
        """Get altitude distribution for this category"""
        return [obj.altitude for obj in self.objects]

class SpacecraftDebris(DebrisCategory):
    """Active and inactive satellites"""
    
    def __init__(self):
        super().__init__(
            name="Spacecraft",
            color="#2E86AB",
            description="Active and inactive satellites"
        )
    
    def generate_distribution(self, count: int) -> List[DebrisObject]:
        """Generate realistic distribution of spacecraft debris"""
        objects = []
        
        # Common satellite altitude bands
        leo_satellites = int(count * 0.7)      # 70% in LEO
        meo_satellites = int(count * 0.15)     # 15% in MEO  
        geo_satellites = int(count * 0.15)     # 15% in GEO
        
        # LEO satellites (300-1200km)
        for _ in range(leo_satellites):
            altitude = np.random.normal(550, 200)
            altitude = max(300, min(1200, altitude))
            
            obj = DebrisObject(
                object_type="spacecraft",
                altitude=altitude,
                size_category="large",
                mass=np.random.uniform(100, 6000),
                orbital_period=self._calculate_period(altitude),
                inclination=np.random.uniform(0, 180)
            )
            objects.append(obj)
        
        # MEO satellites (mostly GPS, around 20,200km)
        for _ in range(meo_satellites):
            altitude = np.random.normal(20200, 500)
            
            obj = DebrisObject(
                object_type="spacecraft",
                altitude=altitude,
                size_category="large",
                mass=np.random.uniform(500, 2000),
                orbital_period=self._calculate_period(altitude),
                inclination=np.random.uniform(50, 65)
            )
            objects.append(obj)
        
        # GEO satellites (35,786km)
        for _ in range(geo_satellites):
            altitude = np.random.normal(35786, 100)
            
            obj = DebrisObject(
                object_type="spacecraft",
                altitude=altitude,
                size_category="large",
                mass=np.random.uniform(1000, 5000),
                orbital_period=24.0,  # Geostationary
                inclination=np.random.uniform(0, 15)
            )
            objects.append(obj)
        
        self.objects = objects
        return objects
    
    def _calculate_period(self, altitude: float) -> float:
        """Calculate orbital period in hours"""
        earth_radius = 6371
        mu = 398600.4418
        
        semi_major_axis = earth_radius + altitude
        period = 2 * np.pi * np.sqrt(semi_major_axis**3 / mu)
        return period / 3600

class RocketBodyDebris(DebrisCategory):
    """Spent rocket upper stages"""
    
    def __init__(self):
        super().__init__(
            name="Rocket Bodies",
            color="#A23B72", 
            description="Spent rocket upper stages and boosters"
        )
    
    def generate_distribution(self, count: int) -> List[DebrisObject]:
        """Generate realistic distribution of rocket body debris"""
        objects = []
        
        for _ in range(count):
            # Most rocket bodies end up in LEO transfer orbits
            if np.random.random() < 0.8:  # 80% in LEO
                altitude = np.random.uniform(200, 2000)
            else:  # 20% in higher orbits
                altitude = np.random.uniform(2000, 35786)
            
            obj = DebrisObject(
                object_type="rocket_body",
                altitude=altitude,
                size_category="large",
                mass=np.random.uniform(500, 8000),
                orbital_period=self._calculate_period(altitude),
                inclination=np.random.uniform(0, 180)
            )
            objects.append(obj)
        
        self.objects = objects
        return objects
    
    def _calculate_period(self, altitude: float) -> float:
        """Calculate orbital period in hours"""
        earth_radius = 6371
        mu = 398600.4418
        
        semi_major_axis = earth_radius + altitude
        period = 2 * np.pi * np.sqrt(semi_major_axis**3 / mu)
        return period / 3600

class MissionDebris(DebrisCategory):
    """Objects released during space missions"""
    
    def __init__(self):
        super().__init__(
            name="Mission Debris",
            color="#F18F01",
            description="Objects released during missions (lens caps, bolts, etc.)"
        )
    
    def generate_distribution(self, count: int) -> List[DebrisObject]:
        """Generate realistic distribution of mission debris"""
        objects = []
        
        for _ in range(count):
            # Mission debris follows spacecraft deployment patterns
            altitude = np.random.choice([
                np.random.uniform(300, 600),    # LEO missions
                np.random.uniform(35786, 35786) # GEO missions  
            ], p=[0.85, 0.15])
            
            obj = DebrisObject(
                object_type="mission_debris",
                altitude=altitude,
                size_category="medium",
                mass=np.random.uniform(0.1, 50),
                orbital_period=self._calculate_period(altitude),
                inclination=np.random.uniform(0, 180)
            )
            objects.append(obj)
        
        self.objects = objects
        return objects
    
    def _calculate_period(self, altitude: float) -> float:
        """Calculate orbital period in hours"""
        earth_radius = 6371
        mu = 398600.4418
        
        semi_major_axis = earth_radius + altitude
        period = 2 * np.pi * np.sqrt(semi_major_axis**3 / mu)
        return period / 3600

class FragmentationDebris(DebrisCategory):
    """Fragments from explosions and collisions"""
    
    def __init__(self):
        super().__init__(
            name="Fragmentation Debris", 
            color="#C73E1D",
            description="Fragments from satellite breakups and collisions"
        )
    
    def generate_distribution(self, count: int) -> List[DebrisObject]:
        """Generate realistic distribution of fragmentation debris"""
        objects = []
        
        # Major breakup altitudes (based on historical events)
        breakup_altitudes = [485, 789, 865, 950, 1200]  # Historical collision/explosion altitudes
        
        for _ in range(count):
            # 60% from major breakup events, 40% distributed
            if np.random.random() < 0.6:
                # Concentrated around major breakup events
                base_altitude = np.random.choice(breakup_altitudes)
                altitude = np.random.normal(base_altitude, 50)
                altitude = max(200, altitude)
            else:
                # Distributed throughout LEO
                altitude = np.random.uniform(200, 2000)
            
            obj = DebrisObject(
                object_type="fragmentation_debris",
                altitude=altitude,
                size_category="small",
                mass=np.random.uniform(0.001, 10),
                orbital_period=self._calculate_period(altitude),
                inclination=np.random.uniform(0, 180)
            )
            objects.append(obj)
        
        self.objects = objects
        return objects
    
    def _calculate_period(self, altitude: float) -> float:
        """Calculate orbital period in hours"""
        earth_radius = 6371
        mu = 398600.4418
        
        semi_major_axis = earth_radius + altitude
        period = 2 * np.pi * np.sqrt(semi_major_axis**3 / mu)
        return period / 3600

# Factory function to create all debris categories
def create_debris_categories() -> List[DebrisCategory]:
    """Create and return all debris category objects"""
    return [
        SpacecraftDebris(),
        RocketBodyDebris(), 
        MissionDebris(),
        FragmentationDebris()
    ]

# Debris size categories
SIZE_CATEGORIES = {
    'large': {'min_size': 100, 'max_size': 10000, 'description': '>1m diameter'},
    'medium': {'min_size': 10, 'max_size': 100, 'description': '10cm-1m diameter'},
    'small': {'min_size': 1, 'max_size': 10, 'description': '1-10cm diameter'},
    'tiny': {'min_size': 0.1, 'max_size': 1, 'description': '1mm-1cm diameter'}
}
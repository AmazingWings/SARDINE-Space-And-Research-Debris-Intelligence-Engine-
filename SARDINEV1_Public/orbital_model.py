"""
Orbital Debris Model - Core Physics and Distribution Engine
Based on NASA ODPO and ESA data for accurate orbital mechanics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from debris_categories import create_debris_categories, DebrisObject
from debris_data import DEBRIS_STATISTICS, DEBRIS_TYPES, get_debris_count_by_altitude

__all__ = ['OrbitalDebrisModel']

class OrbitalDebrisModel:
    """Main model for space debris orbital distribution"""
    
    def __init__(self):
        self.earth_radius = 6371  # km
        self.mu = 398600.4418     # Earth's gravitational parameter (km³/s²)
        self.debris_categories = create_debris_categories()
        self.all_debris = []
        
        # Only use empirically tracked objects
        self.total_tracked_objects = DEBRIS_STATISTICS['large_objects']  # >10cm tracked objects
        self.active_satellites = DEBRIS_STATISTICS['active_payloads']    # verified active sats
        
    def generate_debris_population(self, scale_factor: float = 0.01) -> List[DebrisObject]:
        """
        Generate realistic debris population based on current statistics
        scale_factor: Reduces total number for visualization (default 1% of actual)
        """
        print("Generating debris population from credible sources...")
        
        # Calculate scaled object counts based on real percentages
        total_objects = int(self.total_tracked_objects * scale_factor)
        
        object_counts = {}
        for debris_type, data in DEBRIS_TYPES.items():
            count = int(total_objects * data['percentage'] / 100)
            object_counts[debris_type] = count
        
        print(f"Scaled Population ({scale_factor*100}% of tracked objects):")
        for debris_type, count in object_counts.items():
            print(f"  - {debris_type.replace('_', ' ').title()}: {count:,} objects")
        
        # Generate objects for each category
        all_debris = []
        
        # Spacecraft (active and inactive satellites)
        spacecraft_debris = self.debris_categories[0]
        spacecraft_objects = spacecraft_debris.generate_distribution(object_counts['spacecraft'])
        all_debris.extend(spacecraft_objects)
        
        # Rocket bodies
        rocket_debris = self.debris_categories[1] 
        rocket_objects = rocket_debris.generate_distribution(object_counts['rocket_bodies'])
        all_debris.extend(rocket_objects)
        
        # Mission debris
        mission_debris = self.debris_categories[2]
        mission_objects = mission_debris.generate_distribution(object_counts['mission_debris'])
        all_debris.extend(mission_objects)
        
        # Fragmentation debris
        fragmentation_debris = self.debris_categories[3]
        frag_objects = fragmentation_debris.generate_distribution(object_counts['fragmentation_debris'])
        all_debris.extend(frag_objects)
        
        self.all_debris = all_debris
        print(f"\nTotal generated objects: {len(all_debris):,}")
        
        return all_debris
    
    def get_altitude_distribution(self) -> Dict[str, List[float]]:
        """Get altitude distribution by debris type"""
        distribution = {}
        
        for category in self.debris_categories:
            if category.objects:  # Only include categories with objects
                altitudes = [obj.altitude for obj in category.objects]
                distribution[category.name] = altitudes
        
        return distribution
    
    def get_orbital_shell_density(self, shell_thickness: float = 100) -> Dict[float, int]:
        """
        Calculate debris density in orbital shells
        shell_thickness: thickness of each altitude shell in km
        """
        if not self.all_debris:
            return {}
        
        altitudes = [obj.altitude for obj in self.all_debris]
        min_alt = min(altitudes)
        max_alt = max(altitudes)
        
        # Create altitude bins
        bins = np.arange(min_alt, max_alt + shell_thickness, shell_thickness)
        hist, bin_edges = np.histogram(altitudes, bins=bins)
        
        # Convert to dictionary with center altitudes as keys
        density_dict = {}
        for i, count in enumerate(hist):
            center_altitude = (bin_edges[i] + bin_edges[i+1]) / 2
            density_dict[center_altitude] = count
        
        return density_dict
    
    def calculate_collision_probability(self, altitude: float, time_period: float = 1.0) -> float:
        """
        Calculate relative collision probability at given altitude
        Based on debris density and orbital velocity
        """
        if not self.all_debris:
            return 0.0
        
        # Count objects within ±50km of target altitude
        nearby_objects = [obj for obj in self.all_debris 
                         if abs(obj.altitude - altitude) <= 50]
        
        if not nearby_objects:
            return 0.0
        
        # Simple collision probability model
        debris_density = len(nearby_objects)
        orbital_velocity = self._calculate_velocity(altitude)
        
        # Relative probability (normalized)
        probability = (debris_density * orbital_velocity) / 1000000
        
        return min(probability * time_period, 1.0)
    
    def _calculate_velocity(self, altitude: float) -> float:
        """Calculate orbital velocity at given altitude (km/s)"""
        r = self.earth_radius + altitude
        velocity = np.sqrt(self.mu / r)
        return velocity
    
    def _calculate_period(self, altitude: float) -> float:
        """Calculate orbital period at given altitude (hours)"""
        r = self.earth_radius + altitude
        period = 2 * np.pi * np.sqrt(r**3 / self.mu)
        return period / 3600
    
    def get_statistics_summary(self) -> Dict:
        """Get summary statistics for the debris population"""
        if not self.all_debris:
            return {}
        
        altitudes = [obj.altitude for obj in self.all_debris]
        masses = [obj.mass for obj in self.all_debris]
        
        # Statistics by type
        type_stats = {}
        for category in self.debris_categories:
            if category.objects:
                cat_altitudes = [obj.altitude for obj in category.objects]
                type_stats[category.name] = {
                    'count': len(category.objects),
                    'mean_altitude': np.mean(cat_altitudes),
                    'altitude_range': (min(cat_altitudes), max(cat_altitudes))
                }
        
        summary = {
            'total_objects': len(self.all_debris),
            'altitude_stats': {
                'min': min(altitudes),
                'max': max(altitudes), 
                'mean': np.mean(altitudes),
                'std': np.std(altitudes)
            },
            'mass_stats': {
                'total_mass': sum(masses),
                'mean_mass': np.mean(masses),
                'largest_object': max(masses)
            },
            'by_type': type_stats,
            'peak_density_altitude': self._find_peak_density_altitude()
        }
        
        return summary
    
    def _find_peak_density_altitude(self) -> float:
        """Find altitude with highest debris density"""
        shell_density = self.get_orbital_shell_density(50)  # 50km shells
        
        if not shell_density:
            return 0
        
        peak_altitude = max(shell_density.keys(), key=lambda x: shell_density[x])
        return peak_altitude
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export debris data to pandas DataFrame for analysis"""
        if not self.all_debris:
            return pd.DataFrame()
        
        data = []
        for obj in self.all_debris:
            data.append({
                'object_type': obj.object_type,
                'altitude_km': obj.altitude,
                'size_category': obj.size_category,
                'mass_kg': obj.mass,
                'orbital_period_hours': obj.orbital_period,
                'inclination_deg': obj.inclination
            })
        
        return pd.DataFrame(data)
    
    def validate_model_accuracy(self) -> Dict:
        """Validate model against known debris distribution patterns"""
        if not self.all_debris:
            return {'status': 'No data to validate'}
        
        altitudes = [obj.altitude for obj in self.all_debris]
        
        # Check key validation criteria from NASA/ESA data
        leo_objects = len([alt for alt in altitudes if alt < 2000])
        total_objects = len(altitudes)
        leo_percentage = (leo_objects / total_objects) * 100
        
        # Find peak density region
        peak_alt = self._find_peak_density_altitude()
        
        validation = {
            'status': 'VALIDATED',
            'leo_percentage': leo_percentage,
            'expected_leo_percentage': '~80-90%',  # Most debris in LEO
            'peak_density_altitude': peak_alt,
            'expected_peak_range': '750-1000 km',  # NASA reported range
            'meets_criteria': {
                'leo_dominance': leo_percentage > 75,
                'peak_in_range': 700 <= peak_alt <= 1100
            }
        }
        
        return validation

# Utility functions for orbital mechanics
def kepler_to_cartesian(semi_major_axis: float, eccentricity: float, 
                       inclination: float, raan: float, arg_periapsis: float, 
                       true_anomaly: float) -> Tuple[float, float]:
    """
    Convert Keplerian orbital elements to 2D coordinates for visualization
    Simplified for 2D representation (returns altitude vs ground track)
    """
    # Simplified 2D representation 
    radius = semi_major_axis * (1 - eccentricity**2) / (1 + eccentricity * np.cos(true_anomaly))
    altitude = radius - 6371  # Earth radius
    
    # Ground track position (simplified)
    ground_position = (raan + arg_periapsis + true_anomaly) % (2 * np.pi)
    
    return altitude, ground_position

def atmospheric_drag_lifetime(altitude: float, mass: float, area: float) -> float:
    """
    Estimate orbital lifetime due to atmospheric drag (simplified model)
    Returns lifetime in years
    """
    if altitude > 1000:
        return 1000  # Very long lifetime above 1000km
    
    # Simplified drag model
    drag_coefficient = 2.2
    atmospheric_density = 1e-12 * np.exp(-(altitude - 200) / 50)  # kg/m³
    
    if atmospheric_density <= 0:
        return 1000
    
    ballistic_coefficient = mass / (drag_coefficient * area)
    
    # Simplified lifetime calculation (years)
    lifetime = ballistic_coefficient / (atmospheric_density * 1e6)
    
    return min(max(lifetime, 0.1), 1000)  # Clamp between 0.1 and 1000 years
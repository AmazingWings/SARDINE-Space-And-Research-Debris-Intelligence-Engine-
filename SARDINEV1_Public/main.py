#!/usr/bin/env python3
"""
Space Debris Orbital Distribution Model
Main execution script

Based on credible sources:
- NASA Orbital Debris Program Office (ODPO) 2024 Reports
- ESA Space Environment Report 2025
- NASA Technical Reports Server

Author: Space Debris Research Team
Version: 1.0
Date: September 2024
"""

import sys
import os
import time
import matplotlib.pyplot as plt
from orbital_model import OrbitalDebrisModel
from visualization_new import DebrisVisualizer
from debris_data import DATA_SOURCES, DEBRIS_STATISTICS

def print_header():
    """Print program header and information"""
    print("=" * 80)
    print("🛰️  SPACE DEBRIS ORBITAL DISTRIBUTION MODEL")
    print("=" * 80)
    print("Based on credible sources from NASA ODPO and ESA (2024-2025)")
    print(f"Primary Data: {DATA_SOURCES['primary']}")
    print(f"Secondary Data: {DATA_SOURCES['secondary']}")
    print(f"Last Updated: {DATA_SOURCES['last_updated']}")
    print("=" * 80)
    print()

def print_current_statistics():
    """Print current debris statistics"""
    print("📊 CURRENT SPACE DEBRIS STATISTICS (2024-2025):")
    print("-" * 50)
    print(f"• Tracked Objects (>10cm):     {DEBRIS_STATISTICS['large_objects']:,}")
    print(f"• Medium Objects (1-10cm):     {DEBRIS_STATISTICS['medium_objects']:,}")  
    print(f"• Small Objects (1mm-1cm):     {DEBRIS_STATISTICS['small_objects']:,}")
    print(f"• Active Satellites:           {DEBRIS_STATISTICS['active_payloads']:,}")
    
    # Calculate total excluding the 'sources' dictionary
    total = (DEBRIS_STATISTICS['large_objects'] + 
            DEBRIS_STATISTICS['medium_objects'] + 
            DEBRIS_STATISTICS['small_objects'] + 
            DEBRIS_STATISTICS['active_payloads'])
    print(f"• Total Estimated Objects:     {total:,}")
    print()
    
    # Show data sources
    print("📋 DATA SOURCES:")
    print("-" * 50)
    for key, source in DEBRIS_STATISTICS['sources'].items():
        print(f"• {key.title()}: {source}")
    print()

def get_user_preferences():
    """Get user preferences for model parameters"""
    print("🔧 MODEL CONFIGURATION:")
    print("-" * 30)
    
    # Scale factor (what percentage of actual debris to model)
    # Accepts values from 0.01 (1%) up to 1.0 (100%) for full-scale runs
    while True:
        try:
            scale_input = input("Enter scale factor (0.01 = 1%, 1.0 = 100%) [default: 0.01]: ").strip()
            if not scale_input:
                scale_factor = 0.01
                break
            scale_factor = float(scale_input)
            if 0.01 <= scale_factor <= 1.0:
                break
            else:
                print("⚠️  Scale factor must be between 0.01 (1%) and 1.0 (100%)")
        except ValueError:
            print("⚠️  Please enter a valid number (e.g. 0.01 for 1% or 1.0 for 100%)")
    
    # Visualization options
    print("\n📊 VISUALIZATION OPTIONS:")
    print("1. Show plots interactively")
    print("2. Save plots to files")
    print("3. Both (show and save)")
    
    while True:
        try:
            viz_choice = input("Choose visualization option [1-3, default: 3]: ").strip()
            if not viz_choice:
                viz_choice = 3
                break
            viz_choice = int(viz_choice)
            if 1 <= viz_choice <= 3:
                break
            else:
                print("⚠️  Please choose 1, 2, or 3")
        except ValueError:
            print("⚠️  Please enter a valid number")
    
    return scale_factor, viz_choice

def run_model(scale_factor=0.01):
    """Run the debris model with given parameters"""
    print(f"\n🚀 INITIALIZING DEBRIS MODEL (Scale: {scale_factor*100}%)...")
    print("-" * 50)
    
    # Create model instance
    model = OrbitalDebrisModel()
    
    # Generate debris population
    start_time = time.time()
    debris_objects = model.generate_debris_population(scale_factor=scale_factor)
    generation_time = time.time() - start_time
    
    print(f"✅ Generated {len(debris_objects):,} debris objects in {generation_time:.2f} seconds")
    
    # Get model statistics
    stats = model.get_statistics_summary()
    
    print(f"\n📈 MODEL STATISTICS:")
    print("-" * 30)
    print(f"• Total Objects: {stats['total_objects']:,}")
    print(f"• Altitude Range: {stats['altitude_stats']['min']:.0f} - {stats['altitude_stats']['max']:.0f} km")
    print(f"• Mean Altitude: {stats['altitude_stats']['mean']:.0f} km")
    print(f"• Peak Density Altitude: {stats['peak_density_altitude']:.0f} km")
    print(f"• Total Mass: {stats['mass_stats']['total_mass']:.0f} kg")
    
    # Validate model accuracy
    validation = model.validate_model_accuracy()
    print(f"\n✅ MODEL VALIDATION: {validation['status']}")
    print(f"• LEO Objects: {validation['leo_percentage']:.1f}% (Expected: ~80-90%)")
    print(f"• Peak Density: {validation['peak_density_altitude']:.0f} km (Expected: 750-1000 km)")
    
    if all(validation['meets_criteria'].values()):
        print("🎯 Model meets all validation criteria!")
    else:
        print("⚠️  Some validation criteria not met - check model parameters")
    
    return model

def create_visualizations(model, viz_choice):
    """Create and display/save visualizations"""
    print(f"\n🎨 CREATING VISUALIZATIONS...")
    print("-" * 40)
    
    # Create visualizer
    visualizer = DebrisVisualizer(model)
    
    start_time = time.time()
    
    if viz_choice in [1, 3]:  # Show plots
        print("📊 Generating interactive plots...")
        
        # Create main distribution plot
        fig1 = visualizer.create_orbital_distribution_plot()
        
        # Create detailed analysis plot
        fig2 = visualizer.create_detailed_analysis_plot()
        
        # Create legend reference
        fig3 = visualizer.create_legend_reference()
        
        print("🖼️  Displaying plots (close each window to continue)...")
        plt.show()
    
    if viz_choice in [2, 3]:  # Save plots
        print("💾 Saving plots to files...")
        saved_files = visualizer.save_all_plots("debris_analysis_output")
        
        print("✅ Plots saved successfully:")
        for filename in saved_files:
            print(f"   • {filename}")
    
    viz_time = time.time() - start_time
    print(f"🎯 Visualization complete in {viz_time:.2f} seconds")

def export_data(model):
    """Export model data for further analysis"""
    print(f"\n💾 EXPORTING DATA...")
    print("-" * 25)
    
    # Export to CSV
    df = model.export_to_dataframe()
    output_file = "debris_analysis_output/space_debris_data.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs("debris_analysis_output", exist_ok=True)
    
    df.to_csv(output_file, index=False)
    print(f"📊 Data exported to: {output_file}")
    print(f"   Rows: {len(df):,}, Columns: {len(df.columns)}")
    
    # Print data preview
    print(f"\n📋 DATA PREVIEW:")
    print(df.head().to_string(index=False))
    
    return output_file

def print_summary(model, execution_time):
    """Print execution summary"""
    stats = model.get_statistics_summary()
    
    print(f"\n" + "=" * 80)
    print("📋 EXECUTION SUMMARY")
    print("=" * 80)
    # Use authoritative large_objects count from debris_data for percent calculation
    from debris_data import DEBRIS_STATISTICS
    large_objects_count = DEBRIS_STATISTICS.get('large_objects', 35000)
    percent_of_actual = (len(model.all_debris) / large_objects_count) * 100 if large_objects_count else 0
    print(f"• Model Scale Factor: {percent_of_actual:.2f}% of actual debris")
    print(f"• Objects Generated: {len(model.all_debris):,}")
    print(f"• Peak Debris Altitude: {stats['peak_density_altitude']:.0f} km")
    print(f"• Total Execution Time: {execution_time:.2f} seconds")
    print(f"• Output Directory: ./debris_analysis_output/")
    print("\n🎯 Model successfully demonstrates orbital debris distribution")
    print("   based on credible NASA ODPO and ESA data sources.")
    print("=" * 80)

def main():
    """Main execution function"""
    try:
        # Print header and information
        print_header()
        print_current_statistics()
        
        # Get user preferences
        scale_factor, viz_choice = get_user_preferences()
        
        # Record start time
        start_time = time.time()
        
        # Run the model
        model = run_model(scale_factor)
        
        # Create visualizations
        create_visualizations(model, viz_choice)
        
        # Export data
        export_data(model)
        
        # Calculate total execution time
        total_time = time.time() - start_time
        
        # Print summary
        print_summary(model, total_time)
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Program interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(f"   Please check your Python environment and dependencies")
        sys.exit(1)

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = ['matplotlib', 'numpy', 'pandas', 'seaborn', 'scipy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print(f"Please install with: pip install {' '.join(missing_packages)}")
        return False
    
    return True

if __name__ == "__main__":
    print("🔍 Checking dependencies...")
    
    if not check_dependencies():
        print("💡 Install requirements with: pip install -r requirements.txt")
        sys.exit(1)
    
    print("✅ All dependencies found!")
    print()  # Add a blank line for better formatting
    main()
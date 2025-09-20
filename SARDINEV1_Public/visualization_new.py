"""Space Debris Visualization Module
Creates clean, professional visualizations of orbital debris data.
"""

import os
from typing import Tuple, List

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter, ScalarFormatter

from orbital_model import OrbitalDebrisModel
from debris_data import DATA_SOURCES

class DebrisVisualizer:
    def __init__(self, model: OrbitalDebrisModel):
        self.model = model
        self.earth_radius = 6371
        
        # Set up clean, professional plot style with high contrast
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'figure.figsize': (24, 16),  # Larger figure size
            'figure.dpi': 200,  # Higher DPI for clarity
            'figure.facecolor': 'white',
            'axes.facecolor': '#F8F9FA',  # Light gray background
            'axes.grid': True,
            'grid.alpha': 0.4,  # More visible grid
            'axes.labelsize': 16,  # Larger axis labels
            'axes.titlesize': 20,  # Larger titles
            'xtick.labelsize': 14,  # Larger tick labels
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'font.weight': 'bold',  # Bolder fonts
            'axes.labelweight': 'bold',
            'axes.titleweight': 'bold',
            'figure.titleweight': 'bold',
            'axes.spines.top': False,  # Cleaner look
            'axes.spines.right': False,
            'grid.linestyle': '--',  # Dashed grid lines
            'grid.linewidth': 0.8
        })
        
        # Enhanced color scheme with better contrast
        self.colors = {
            "earth": "#2E86DE",  # Deeper blue
            "atmosphere": "#54A0FF",  # Brighter blue
            "spacecraft": "#FF6B6B",  # Red
            "rocket": "#1DD1A1",  # Green
            "debris": "#A55EEA",  # Purple
            "inactive": "#576574"  # Gray
        }

    def create_detailed_analysis_plot(self, figsize: Tuple[int, int] = (24, 16)) -> plt.Figure:
        fig = plt.figure(figsize=figsize)
        
        # Improved spacing with larger plots and better margins
        ax1 = fig.add_axes([0.08, 0.55, 0.40, 0.38])  # Mass Distribution
        ax2 = fig.add_axes([0.56, 0.55, 0.40, 0.38])  # Orbital Period
        ax3 = fig.add_axes([0.08, 0.08, 0.40, 0.38])  # Collision Risk
        ax4 = fig.add_axes([0.56, 0.08, 0.40, 0.38])  # Statistics
        
        # Enhanced title with more space
        fig.text(0.5, 0.96, "Space Debris Analysis", 
                fontsize=24, weight='bold', ha='center')
        
        self._plot_mass_distribution(ax1)
        self._plot_orbital_period_distribution(ax2)
        self._plot_collision_risk(ax3)
        self._plot_statistics_summary(ax4)
        
        return fig

    def _plot_mass_distribution(self, ax):
        for cat in self.model.debris_categories:
            if getattr(cat, "objects", None):
                masses = [o.mass for o in cat.objects]
                # Use better bins and transparency
                n_bins = np.ceil(np.sqrt(len(masses)))
                ax.hist(masses, bins=int(n_bins), alpha=0.6, label=cat.name,
                       color=self.colors.get(cat.name.lower(), "#999999"),
                       density=True, edgecolor='white', linewidth=1)
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Object Mass (kilograms)", fontsize=16, labelpad=15)
        ax.set_ylabel("Relative Frequency", fontsize=16, labelpad=15)
        ax.set_title("Mass Distribution of Space Objects", pad=20, fontsize=20)
        
        # Add grid and format axes
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        
        # Add mass range annotation
        if self.model.all_debris:
            masses = [o.mass for o in self.model.all_debris]
            ax.text(0.02, 0.98, 
                   f"Mass Range:\n{min(masses):,.2f} - {max(masses):,.2f} kg",
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
                   va='top')
        
        # Move legend outside plot
        ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=14,
                 borderaxespad=0, frameon=True, fancybox=True, shadow=True)

    def _plot_orbital_period_distribution(self, ax):
        periods = [o.orbital_period for o in getattr(self.model, "all_debris", [])
                  if getattr(o, "orbital_period", None) is not None and o.orbital_period < 50]
        
        if not periods:
            ax.text(0.5, 0.5, "No period data", ha="center", va="center")
            return
        
        # Create histogram with better styling
        n, bins, patches = ax.hist(periods, bins=50, color=self.colors["earth"],
                                 alpha=0.7, edgecolor='white', linewidth=1)
        
        # Add improved reference lines and regions
        max_height = max(n)
        references = [
            (1.5, "Low Earth Orbit (LEO)", self.colors["spacecraft"]),
            (12, "Medium Earth Orbit (MEO)", self.colors["rocket"]),
            (24, "Geosynchronous Orbit (GEO)", self.colors["debris"])
        ]
        
        for period, label, color in references:
            ax.axvline(period, color=color, ls='--', alpha=0.8)
            ax.text(period + 0.5, max_height * 0.95, label,
                   color=color, fontsize=12)
        
        ax.set_xlabel("Orbital Period (hours)", fontsize=14, labelpad=10)
        ax.set_ylabel("Number of Objects", fontsize=14, labelpad=10)
        ax.set_title("Orbital Period Distribution", pad=20, fontsize=16)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_xlim(0, 30)

    def _plot_collision_risk(self, ax):
        alts = np.linspace(200, 2000, 100)
        risks = [self.model.calculate_collision_probability(a) for a in alts]
        
        ax.plot(alts, risks, color='#FF4757', linewidth=2)
        ax.fill_between(alts, risks, alpha=0.2, color='#FF4757')
        
        ax.set_xlabel("Altitude (km)", fontsize=14, labelpad=10)
        ax.set_ylabel("Relative Collision Risk", fontsize=14, labelpad=10)
        ax.set_title("Collision Risk by Altitude", pad=20, fontsize=16)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

    def _plot_statistics_summary(self, ax):
        stats = self.model.get_statistics_summary()
        if not stats:
            ax.text(0.5, 0.5, "No statistics available", ha="center", va="center")
            return

        def format_number(n):
            if n >= 1_000_000:
                return f"{n/1_000_000:.1f}M"
            elif n >= 1_000:
                return f"{n/1_000:.1f}K"
            return f"{n:.0f}"

        lines = [
            "MODEL STATISTICS",
            "─" * 30,
            f"Total Objects: {format_number(stats['total_objects'])}",
            "",
            "ALTITUDE STATS",
            f"Range: {stats['altitude_stats']['min']:.0f} - {stats['altitude_stats']['max']:.0f} km",
            f"Mean:  {stats['altitude_stats']['mean']:.0f} km",
            "",
            "MASS STATS",
            f"Total: {format_number(stats['mass_stats']['total_mass'])} kg",
            "",
            "CATEGORIES"
        ]

        for name, t in stats.get('by_type', {}).items():
            lines.append(f"{name}: {format_number(t.get('count', 0))}")

        ax.text(0.05, 0.95, "\n".join(lines),
                va="top", fontsize=12, family="monospace",
                bbox=dict(facecolor='white', edgecolor='#DDD',
                         boxstyle='round,pad=1'))
        ax.axis('off')

    def _plot_2d_orbital_view(self, ax):
        if not getattr(self.model, "all_debris", None):
            ax.text(0.5, 0.5, "No debris data available", ha="center", va="center")
            return

        # Generate points
        n = len(self.model.all_debris)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        np.random.shuffle(angles)

        # Plot each category
        start = 0
        max_alt = 0
        for cat in self.model.debris_categories:
            if not getattr(cat, "objects", None):
                continue
            count = len(cat.objects)
            alts = [o.altitude for o in cat.objects]
            if alts:
                max_alt = max(max_alt, max(alts))
            ang = angles[start:start + count]
            x = [(self.earth_radius + a) * np.cos(t) for a, t in zip(alts, ang)]
            y = [(self.earth_radius + a) * np.sin(t) for a, t in zip(alts, ang)]
            ax.scatter(x, y, c=getattr(cat, "color", "#777777"),
                      s=6, alpha=0.45, label=cat.name, linewidths=0)
            start += count

        # Draw Earth and atmosphere
        ax.add_patch(patches.Circle((0, 0), self.earth_radius,
                                  color=self.colors["earth"], zorder=0))
        ax.add_patch(patches.Circle((0, 0), self.earth_radius + 100,
                                  fill=False, color=self.colors["atmosphere"],
                                  ls="--", alpha=0.6))

        # Set limits and style
        lim = max(self.earth_radius + 40000, self.earth_radius + max_alt)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title("2D Orbital Distribution", pad=20, fontsize=16)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)

    def _plot_altitude_histogram(self, ax):
        if not getattr(self.model, "all_debris", None):
            ax.text(0.5, 0.5, "No debris data available", ha="center", va="center")
            return

        # Collect data by type
        data_by_type = {}
        for cat in self.model.debris_categories:
            if getattr(cat, "objects", None):
                data_by_type[cat.name] = [o.altitude for o in cat.objects]

        # Create stacked histogram
        bins = np.logspace(np.log10(200), np.log10(40000), 50)
        bottom = np.zeros(len(bins) - 1)
        
        # Add LEO region highlight
        leo_mask = bins[:-1] < 2000
        max_height = max([len(alts) for alts in data_by_type.values()])
        ax.fill_between(bins[:-1][leo_mask], 0, 
                       [max_height * 1.2] * sum(leo_mask),
                       color='red', alpha=0.1, label='LEO Region')
        
        # Plot stacked bars
        for name, alts in data_by_type.items():
            color = getattr(next((c for c in self.model.debris_categories
                                if c.name == name), None), "color", "#777")
            hist, _ = np.histogram(alts, bins=bins)
            ax.bar(bins[:-1], hist, width=np.diff(bins),
                  bottom=bottom, color=color, alpha=0.75,
                  align="edge", linewidth=0, label=name)
            bottom += hist
            
        # Add annotation showing percentage in LEO
        leo_objects = sum(sum(1 for a in alts if a < 2000) 
                         for alts in data_by_type.values())
        total_objects = sum(len(alts) for alts in data_by_type.values())
        leo_percentage = (leo_objects / total_objects) * 100 if total_objects else 0
        
        ax.text(0.02, 0.98, 
                f"LEO Region (160-2000 km)\n{leo_percentage:.1f}% of all objects",
                transform=ax.transAxes, fontsize=14, weight='bold',
                bbox=dict(facecolor='white', edgecolor='red', alpha=0.9),
                va='top')

        # Style
        ax.set_xscale("log")
        ax.set_xlabel("Altitude (km)", fontsize=14, labelpad=10)
        ax.set_ylabel("Number of Objects", fontsize=14, labelpad=10)
        ax.set_title("Altitude Distribution", pad=20, fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add reference lines
        ax.axvline(400, color="red", ls="--", alpha=0.7)
        ax.axvline(35786, color="green", ls="--", alpha=0.7)

        # Format axis labels
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _:
            f"{int(x):,}" if x >= 1000 else f"{int(x)}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _:
            f"{int(y):,}"))

    def _plot_orbital_shell_density(self, ax):
        shell_density = self.model.get_orbital_shell_density(100)
        if not shell_density:
            ax.text(0.5, 0.5, "No density data available", ha="center", va="center")
            return

        # Plot density curve
        alts = list(shell_density.keys())
        dens = list(shell_density.values())
        ax.plot(alts, dens, marker="o", color="darkblue", linewidth=2)
        ax.fill_between(alts, dens, alpha=0.25, color="lightblue")

        # Style
        ax.set_xlabel("Altitude (km)", fontsize=14, labelpad=10)
        ax.set_ylabel("Objects per 100km Shell", fontsize=14, labelpad=10)
        ax.set_title("Debris Density Distribution", pad=20, fontsize=16)
        ax.grid(True, alpha=0.3)

        # Add peak annotation
        if dens:
            peak_idx = int(np.argmax(dens))
            ax.annotate(f"Peak: {alts[peak_idx]:.0f} km\n{dens[peak_idx]:,} objects",
                       xy=(alts[peak_idx], dens[peak_idx]),
                       xytext=(20, 20), textcoords="offset points",
                       bbox=dict(boxstyle="round,pad=0.5",
                               fc="yellow", alpha=0.85),
                       arrowprops=dict(arrowstyle="->"))

    def _plot_debris_composition(self, ax):
        counts = {cat.name: len(cat.objects)
                 for cat in self.model.debris_categories
                 if getattr(cat, "objects", None)}
        if not counts:
            ax.text(0.5, 0.5, "No composition data available",
                   ha="center", va="center")
            return

        # Create pie chart
        labels = list(counts.keys())
        sizes = list(counts.values())
        colors = [getattr(next((c for c in self.model.debris_categories
                              if c.name == l), None), "color", "#777")
                 for l in labels]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct=lambda p: f"{p:.1f}%",
            startangle=90, textprops={"fontsize": 12}
        )

        # Style
        ax.set_title("Debris Population by Type", pad=20, fontsize=16)
        
        # Add total count
        ax.text(0, -1.2, f"Total Objects: {sum(sizes):,}",
                ha="center", fontsize=12, weight='bold')

    def create_orbital_distribution_plot(self, figsize: Tuple[int, int] = (24, 16)) -> plt.Figure:
        fig = plt.figure(figsize=figsize)
        
        # Improved spacing with larger plots
        ax1 = fig.add_axes([0.08, 0.55, 0.40, 0.38])  # 2D View
        ax2 = fig.add_axes([0.56, 0.55, 0.40, 0.38])  # Altitude Histogram
        ax3 = fig.add_axes([0.08, 0.08, 0.40, 0.38])  # Shell Density
        ax4 = fig.add_axes([0.56, 0.08, 0.40, 0.38])  # Composition
        
        # Enhanced title
        fig.text(0.5, 0.96, "Orbital Debris Distribution Analysis",
                fontsize=24, weight='bold', ha='center')
        
        # Add subtitle with data source
        fig.text(0.5, 0.93, "Based on NASA ODPO and ESA space debris tracking data",
                fontsize=14, style='italic', ha='center', alpha=0.7)
        
        self._plot_2d_orbital_view(ax1)
        self._plot_altitude_histogram(ax2)
        self._plot_orbital_shell_density(ax3)
        self._plot_debris_composition(ax4)
        
        return fig

    def create_legend_reference(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Create a reference figure explaining the visualization elements"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        # Create explanation text
        explanations = [
            ("Orbital Regions", [
                "LEO (Low Earth Orbit): 160-2000 km",
                "MEO (Medium Earth Orbit): 2000-35786 km",
                "GEO (Geosynchronous Orbit): ~35786 km"
            ]),
            ("Object Categories", [
                "Active Spacecraft: Operating satellites and space stations",
                "Rocket Bodies: Upper stages and boosters",
                "Mission Debris: Intentionally released objects",
                "Fragmentation Debris: Results of collisions/explosions"
            ]),
            ("Data Sources", [
                f"Primary: {DATA_SOURCES['primary']}",
                f"Secondary: {DATA_SOURCES['secondary']}",
                "Last Updated: September 2024"
            ])
        ]
        
        y_pos = 0.95
        for title, items in explanations:
            ax.text(0.05, y_pos, title, fontsize=16, weight='bold')
            y_pos -= 0.1
            for item in items:
                ax.text(0.1, y_pos, "• " + item, fontsize=14)
                y_pos -= 0.08
            y_pos -= 0.05
        
        ax.axis('off')
        fig.suptitle("Space Debris Visualization Guide", 
                    fontsize=20, weight='bold', y=0.98)
        
        return fig

    def save_all_plots(self, output_dir: str = "debris_plots"):
        os.makedirs(output_dir, exist_ok=True)
        
        fig1 = self.create_orbital_distribution_plot()
        fig1.savefig(f"{output_dir}/orbital_distribution.png",
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig1)
        
        fig2 = self.create_detailed_analysis_plot()
        fig2.savefig(f"{output_dir}/detailed_analysis.png",
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        
        fig3 = self.create_legend_reference()
        fig3.savefig(f"{output_dir}/legend_reference.png",
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig3)
        
        return [
            f"{output_dir}/orbital_distribution.png",
            f"{output_dir}/detailed_analysis.png",
            f"{output_dir}/legend_reference.png"
        ]
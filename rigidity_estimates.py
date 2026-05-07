import numpy as np
import matplotlib.pyplot as plt

def generate_and_plot_eigenvalues(size=5, is_symmetric=False):
    # 1. Generate a Random Matrix
    if is_symmetric:
        A = np.random.randn(size, size)
        A = (A + A.T) / 2 
    else:
        A = np.random.randn(size, size)

    # 2. Calculate Eigenvalues
    eigenvalues = np.linalg.eigvals(A)
    eigenvalues = np.sort_complex(eigenvalues)

    # 3. Print Results
    print(f"--- Matrix A ({size}x{size}) ---")
    print(np.round(A, 2))
    print("\n--- Calculated Eigenvalues ---")
    for i, val in enumerate(eigenvalues):
        print(f"λ{i+1}: {val:.4f}")

    # 4. Visualize Locations
    plt.figure(figsize=(12, 5))

    # --- PLOT 1: Eigenvalues on Complex Plane ---
    plt.subplot(1, 2, 1)
    plt.scatter(eigenvalues.real, eigenvalues.imag, color='red', label='Eigenvalues', s=50)
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(f'Eigenvalue Locations (Spectrum)')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.legend()
    
    # DYNAMIC ZOOM: Calculate limits based on data with 20% padding
    x_margin = (np.max(np.abs(eigenvalues.real)) + 1) * 1.2
    y_margin = (np.max(np.abs(eigenvalues.imag)) + 1) * 1.2
    plt.xlim(-x_margin, x_margin)
    plt.ylim(-y_margin, y_margin)

    # --- PLOT 2: Gershgorin Circles ---
    plt.subplot(1, 2, 2)
    ax = plt.gca()
    
    # We need to track the max extent of circles to zoom out correctly
    max_circle_extent = 0

    for i in range(size):
        center = A[i, i]
        radius = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        
        # Update max extent for zooming
        extent = np.abs(center) + radius
        if extent > max_circle_extent:
            max_circle_extent = extent

        circle = plt.Circle((center.real, center.imag), radius, fill=False, color='blue', alpha=0.5, linewidth=1.5)
        ax.add_patch(circle)
        plt.plot(center.real, center.imag, 'bx') 

    # Re-plot eigenvalues on top
    plt.scatter(eigenvalues.real, eigenvalues.imag, color='red', zorder=10, s=50)
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(f'Gershgorin Circles (Bounds)')
    plt.xlabel('Real Part')
    
    # DYNAMIC ZOOM for Gershgorin Plot
    # If no circles were drawn (size 0), default to 1. Otherwise use calculated max
    zoom_limit = max(max_circle_extent, 1) * 1.1 
    plt.xlim(-zoom_limit, zoom_limit)
    plt.ylim(-zoom_limit, zoom_limit)

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button, RadioButtons
import cv2
from tkinter import Tk, filedialog
import os
import tempfile

def create_mesh(Xrange, Yrange, Xdiv, Ydiv):
    """Crea la malla para la simulación."""
    x = np.linspace(0, Xrange, Xdiv)
    y = np.linspace(0, Yrange, Ydiv)
    X, Y = np.meshgrid(x, y)
    
    print(f"Mesh dimension: {Xdiv}x{Ydiv}")
    print("--------------------")
    
    return x, y, X, Y

def create_cylinder(Xrange, Yrange, R, dx, dy, Xdiv, Ydiv, x, y):
    """Crea el cilindro y determina qué puntos están dentro de él."""
    # Posiciones para el cilindro
    n = 360
    tetha = np.linspace(0, 2 * np.pi, n)
    center_x, center_y = Xrange/2, Yrange/2
    
    # Coordenadas del contorno del cilindro (alta resolución para visualización)
    Xcylinder = center_x + R * np.cos(tetha)
    Ycylinder = center_y + R * np.sin(tetha)
    
    # Versión ajustada a la malla para los cálculos
    Xaranged = np.round(Xcylinder / dx) * dx
    Yaranged = np.round(Ycylinder / dy) * dy
    
    print("Object contour created")
    print("--------------------")
    
    # Crea un Path del cilindro para determinar puntos interiores
    boundary = Path(np.column_stack([Xaranged, Yaranged]))
    
    # Identifica los puntos dentro del cilindro
    cylinder = np.zeros((Ydiv, Xdiv), dtype=bool)
    for j in range(Ydiv):
        for i in range(Xdiv):
            if boundary.contains_point((x[i], y[j])):
                cylinder[j, i] = True
    
    print("Cylinder mask created")
    print("--------------------")
    
    # También devolvemos las coordenadas de alta resolución para visualización
    return cylinder, Xaranged, Yaranged, Xcylinder, Ycylinder

def create_model(X, Y, dx, dy, Xdiv, Ydiv, x, y):
    """Crea un modelo basado en coordenadas X e Y y determina qué puntos están dentro."""
    
    # Crear un Path para el contorno
    # Asegurarse de que X e Y sean arrays de coordenadas para el contorno
    boundary = Path(np.column_stack([X, Y]))
    
    # Identificar los puntos dentro del modelo
    model = np.zeros((Ydiv, Xdiv), dtype=bool)
    for j in range(Ydiv):
        for i in range(Xdiv):
            if boundary.contains_point((x[i], y[j])):
                model[j, i] = True
    
    print("Object mask created")
    print("--------------------")
    
    # Devolver también las coordenadas originales para visualización
    return model, X, Y

def Airfoil_model(R, Xrange, Yrange, alpha = 0, n_points=500):
    """Genera los puntos para un perfil aerodinámico usando la transformación de Joukowski."""
    # Parámetros del círculo original (antes de transformación)
    R = 1.1                 # Radio del círculo
    center = -0.1 + 0.1j    # Centro desplazado del círculo (a + ib)

    # Puntos en el círculo
    theta = np.linspace(0, 2 * np.pi, n_points)
    z = center + R * np.exp(1j * theta)  # Círculo en el plano complejo

    # Transformación de Joukowski
    def joukowski(z):
        return z + 1/z

    # Aplicar la transformación
    z_airfoil = joukowski(z)

    Xpoints = np.real(z_airfoil)
    Ypoints = np.imag(z_airfoil)
    
    # Escalar y centrar el perfil en el dominio
    x_min, x_max = np.min(Xpoints), np.max(Xpoints)
    y_min, y_max = np.min(Ypoints), np.max(Ypoints)
    
    # Factor de escala para ajustar el tamaño del perfil (más pequeño para mejor resolución)
    scale_factor = 0.6 * min(Xrange, Yrange) / max(x_max - x_min, y_max - y_min)
    
    # Centrar el perfil en el dominio
    Xpoints = (Xpoints - (x_min + x_max)/2) * scale_factor + Xrange/2
    Ypoints = (Ypoints - (y_min + y_max)/2) * scale_factor + Yrange/2

    # Aplicar rotación por ángulo de ataque si está especificado
    if alpha != 0:
        # Punto central alrededor del cual rotar
        center_x, center_y = Xrange/2, Yrange/2
        
        # Trasladar al origen, rotar y volver a la posición
        Xpoints_centered = Xpoints - center_x
        Ypoints_centered = Ypoints - center_y
        
        # Aplicar rotación
        Xpoints_rotated = Xpoints_centered * np.cos(alpha) - Ypoints_centered * np.sin(alpha)
        Ypoints_rotated = Xpoints_centered * np.sin(alpha) + Ypoints_centered * np.cos(alpha)
        
        # Trasladar de vuelta
        Xpoints = Xpoints_rotated + center_x
        Ypoints = Ypoints_rotated + center_y

    
    print("Airfoil geometry created")
    
    return Xpoints, Ypoints

def process_png_image(image_path, target_width, target_height):
    """Procesa una imagen PNG para extraer su contorno y redimensionarla."""
    print(f"Procesando imagen: {image_path}")
    
    # 1. Cargar imagen con canal alfa
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Verificar si la imagen tiene canal alfa
    if img.shape[2] < 4:
        print("La imagen no tiene canal alfa (transparencia). Usando umbral por brillo.")
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        # Usar canal alfa
        alpha = img[:, :, 3]
        _, thresh = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    

    # 2. Encontrar contornos
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No se encontraron contornos en la imagen")
        return None, None
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Usar el contorno con mayor área
    main_contour = max(contours, key=cv2.contourArea)
    
    # 3. Extraer coordenadas originales
    X = main_contour[:, 0, 0].astype(np.float32)
    Y = main_contour[:, 0, 1].astype(np.float32)
    
    # 4. Normalizar al espacio objetivo
    min_x, max_x = X.min(), X.max()
    min_y, max_y = Y.min(), Y.max()
    
    # Escalar manteniendo proporción
    scale_x = target_width / (max_x - min_x)
    scale_y = target_height / (max_y - min_y)
    scale = min(scale_x, scale_y)  # Mantener proporción
    
    # Aplicar escalado y centrado
    X_scaled = (X - min_x) * scale
    Y_scaled = (Y - min_y) * scale
    
    # Centramos en el canvas de tamaño fijo
    X_final = X_scaled + (target_width - X_scaled.max()) / 2 + Xrange/4
    Y_final = Y_scaled + (target_height - Y_scaled.max()) / 2
    
    # Invertir Y para ajustarse al sistema de coordenadas de matplotlib
    Y_final = -Y_final + target_height + Yrange/4
    
    print("Procesamiento de imagen completado")
    
    return X_final, Y_final

def select_png_file():
    """Abre un diálogo para seleccionar un archivo PNG."""
    root = Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter
    
    file_path = filedialog.askopenfilename(
        title="Seleccionar imagen PNG",
        filetypes=[("Archivos PNG", "*.png"), ("Todos los archivos", "*.*")]
    )
    
    root.destroy()
    
    if file_path:
        print(f"Archivo seleccionado: {file_path}")
        return file_path
    else:
        print("No se seleccionó ningún archivo")
        return None

def initialize_stream_function(Uinf, Xdiv, Ydiv, x, y, model, C, Yrange):
    """Inicializa la función de corriente con condiciones de flujo libre."""
    # Crea la matriz de la función de corriente
    ψ = np.zeros((Ydiv, Xdiv))
    
    # Inicializa con condiciones de flujo libre y model
    for j in range(Ydiv):
        for i in range(Xdiv):
            if model[j, i]:
                ψ[j, i] = C  # Valor constante dentro del modelo
            else:
                ψ[j, i] = Uinf * y[j]  # Flujo libre
    
    # Aplica condiciones de contorno en los bordes
    # Borde inferior y superior
    for i in range(Xdiv):
        ψ[0, i] = 0
        ψ[-1, i] = Uinf * Yrange
    
    # Borde izquierdo y derecho
    for j in range(Ydiv):
        ψ[j, 0] = Uinf * y[j]
        ψ[j, -1] = Uinf * y[j]
    
    return ψ

def solve_laplace(ψ, model, Xdiv, Ydiv, iterations):
    """Resuelve la ecuación de Laplace para la función de corriente."""
    print("Starting numerical solution...")
    
    # Itera para resolver la ecuación de Laplace
    for iter in range(iterations):
        ψ_new = np.copy(ψ)
        
        # Actualiza solo los puntos fuera del modelo
        for j in range(1, Ydiv-1):
            for i in range(1, Xdiv-1):
                if not model[j, i]:
                    ψ_new[j, i] = 0.25 * (ψ[j, i+1] + ψ[j, i-1] + ψ[j+1, i] + ψ[j-1, i])
        
        # Actualiza la función de corriente
        ψ = ψ_new
        
        # Reporte de progreso
        if (iter + 1) % (iterations//10) == 0:
            progress = (iter + 1) / iterations * 100
            print(f"{int(progress)}% done")
    
    print("Solution completed")
    
    return ψ

def change_model(label, simulation_data):
    """Cambia entre modelo de perfil aerodinámico y cilindro."""
    # Extraer datos relevantes
    status_text = simulation_data['status_text']
    fig = simulation_data['fig']
    
    # Actualizar texto de estado
    status_text.set_text(f"Cambiando modelo a {label}...")
    
    # Si seleccionamos "Otro Modelo", abrimos un diálogo para seleccionar un archivo PNG
    if label == "Otro Modelo":
        png_file = select_png_file()
        if png_file:
            simulation_data['png_file'] = png_file
            # Procesamos la imagen ahora para obtener los puntos del contorno
            X_png, Y_png = process_png_image(png_file, simulation_data['Xrange']/2, simulation_data['Yrange']/2)
            
            if X_png is not None and Y_png is not None:
                simulation_data['custom_model_coords'] = (X_png, Y_png)
            else:
                status_text.set_text("Error al procesar la imagen. Usando perfil aerodinámico por defecto.")
                label = "Perfil Aerodinámico"
        else:
            # Si no se seleccionó archivo, volvemos al modelo anterior
            status_text.set_text("No se seleccionó archivo. Manteniendo modelo actual.")
            return
    
    # Aquí actualizamos el valor seleccionado en los Radio Buttons
    simulation_data['current_model'] = label
    
    fig.canvas.draw_idle()
    
    update(None, simulation_data)

def update(val, simulation_data):
    """Actualiza la visualización cuando se cambia el valor de C, AoA o dx&dy"""
    # Extraer datos de la simulación del diccionario
    Uinf = simulation_data['Uinf']
    Yrange = simulation_data['Yrange']
    Xdiv = simulation_data['Xdiv']
    Ydiv = simulation_data['Ydiv']
    x = simulation_data['x']
    y = simulation_data['y']
    X = simulation_data['X']
    Y = simulation_data['Y']
    model = simulation_data['model']
    ax = simulation_data['ax']
    fig = simulation_data['fig']
    slider = simulation_data['slider']
    status_text = simulation_data['status_text']
    iterations = simulation_data['iterations']
    info_text = simulation_data['info_text']
    modos = simulation_data['modos']
    
    # Obtener el modelo seleccionado actual
    current_model = simulation_data.get('current_model', modos.value_selected)
    
    Xrange = simulation_data['Xrange']
    R = simulation_data['R']
    dx = simulation_data['dx']
    dy = simulation_data['dy']
    res_slider = simulation_data["res_slider"]
    angle_slider = simulation_data.get("angle_slider", None)

    # Obtener el valor actual de C del deslizador
    C = slider.val

    dx = res_slider.val
    dy = res_slider.val

    # Recalcular la malla con la nueva resolución
    Xdiv = int(Xrange / dx + 1)
    Ydiv = int(Yrange / dy + 1)
    
    x, y, X, Y = create_mesh(Xrange, Yrange, Xdiv, Ydiv)

    # Verificamos el modelo actual correctamente
    if current_model == 'Perfil Aerodinámico':
        # Usar alta resolución para el perfil aerodinámico (visualización)
        Xpoints, Ypoints = Airfoil_model(R, Xrange, Yrange, alpha=np.deg2rad(-angle_slider.val), n_points=500)
        # Usar los mismos puntos para los cálculos y visualización
        model, Xaranged, Yaranged = create_model(Xpoints, Ypoints, dx, dy, Xdiv, Ydiv, x, y)
        # Guardar las coordenadas de alta definición para dibujar
        Xplot, Yplot = Xpoints, Ypoints
    elif current_model == 'Otro Modelo':
        # Verificar si tenemos coordenadas de un modelo personalizado
        if 'custom_model_coords' in simulation_data:
            Xpoints, Ypoints = simulation_data['custom_model_coords']
            model, Xaranged, Yaranged = create_model(Xpoints, Ypoints, dx, dy, Xdiv, Ydiv, x, y)
            Xplot, Yplot = Xpoints, Ypoints
        else:
            # Si no hay coordenadas disponibles, intentar cargar un archivo
            status_text.set_text("No hay modelo personalizado. Seleccione un archivo PNG.")
            png_file = select_png_file()
            if png_file:
                X_png, Y_png = process_png_image(png_file, Xrange/2, Yrange/2)
                if X_png is not None and Y_png is not None:
                    simulation_data['custom_model_coords'] = (X_png, Y_png)
                    Xpoints, Ypoints = X_png, Y_png
                    model, Xaranged, Yaranged = create_model(Xpoints, Ypoints, dx, dy, Xdiv, Ydiv, x, y)
                    Xplot, Yplot = Xpoints, Ypoints
                else:
                    # Si hay error en el procesamiento, usar perfil aerodinámico
                    status_text.set_text("Error al procesar imagen. Usando perfil por defecto.")
                    Xpoints, Ypoints = Airfoil_model(R, Xrange, Yrange, n_points=500)
                    model, Xaranged, Yaranged = create_model(Xpoints, Ypoints, dx, dy, Xdiv, Ydiv, x, y)
                    Xplot, Yplot = Xpoints, Ypoints
            else:
                # Si no se seleccionó archivo, usar perfil aerodinámico
                status_text.set_text("No se seleccionó archivo. Usando perfil por defecto.")
                Xpoints, Ypoints = Airfoil_model(R, Xrange, Yrange, n_points=500)
                model, Xaranged, Yaranged = create_model(Xpoints, Ypoints, dx, dy, Xdiv, Ydiv, x, y)
                Xplot, Yplot = Xpoints, Ypoints
    else:  # Cilindro con alta definición para visualización
        # Obtener máscara para cálculos y coordenadas de alta definición para visualización
        model, Xaranged, Yaranged, Xcylinder_HD, Ycylinder_HD = create_cylinder(
            Xrange, Yrange, 0.3, dx, dy, Xdiv, Ydiv, x, y)
        # Usar las coordenadas de alta definición para dibujar
        Xplot, Yplot = Xcylinder_HD, Ycylinder_HD

    # Actualizar los datos de simulación con la nueva malla
    simulation_data['x'] = x
    simulation_data['y'] = y
    simulation_data['X'] = X
    simulation_data['Y'] = Y
    simulation_data['Xdiv'] = Xdiv
    simulation_data['Ydiv'] = Ydiv
    simulation_data['dx'] = dx
    simulation_data['dy'] = dy
    simulation_data["model"] = model
    simulation_data["Xplot"] = Xplot  # Guardar coordenadas de alta definición para dibujar
    simulation_data["Yplot"] = Yplot
    simulation_data['current_model'] = current_model

    # Actualizar texto de estado
    status_text.set_text(f"Recalculando con dx={dx:.3f}, dy={dy:.3f}, C/UinfYrange = {C:.2f}...")
    fig.canvas.draw_idle()
    
    # Reinicializar la función de corriente
    psi = initialize_stream_function(Uinf, Xdiv, Ydiv, x, y, model, C, Yrange)
    
    # Resolver la ecuación de Laplace
    psi = solve_laplace(psi, model, Xdiv, Ydiv, iterations)
    
    # Limpiar los contornos anteriores
    ax.clear()
    
    # Dibujar los nuevos contornos
    contour = ax.contour(X, Y, psi, levels=40, cmap='inferno')
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Preparar el título con información de ángulo si es relevante
    title = f"Líneas de Corriente - {current_model} (C = {C:.2f})"
    if angle_slider and current_model == 'Perfil Aerodinámico':
        title += f", Ángulo = {angle_slider.val}°"
    
    ax.set_title(title)
    
    # Dibujar el modelo usando las coordenadas de alta definición
    # Usamos las coordenadas guardadas en Xplot y Yplot que son de alta definición
    ax.plot(Xplot, Yplot, 'k-', linewidth=2)
    
    ax.grid(True)
    
    # Añadir nueva colorbar
    if 'cbar' not in simulation_data or not simulation_data["cbar"]:
        cbar = fig.colorbar(contour, ax=ax, label='ψ(x,y)')
        simulation_data['cbar'] = cbar
    
    # Calcular y mostrar la sustentación teórica
    circulation = 2 * np.pi * (C - 0.5) * Uinf * Yrange
    lift = 1.0 * Uinf * circulation  # ρ=1 por simplicidad
    
    if lift > 0:
        lift_label = "Sustentación positiva"
    elif lift < 0:
        lift_label = "Sustentación negativa"
    else:
        lift_label = "Sin sustentación"
    
    info_text.set_text(f'C = {C:.2f} ({"Sin sustentación" if abs(C - 0.5 * Uinf * Yrange) < 0.01 else lift_label})')
    
    # Añadir texto con información de la sustentación
    ax.text(
        0.02, 0.98, 
        f"Circulación: {circulation:.2f}\n{lift_label}",
        transform=ax.transAxes, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    # Actualizar texto de estado
    status_text.set_text("Listo.")
    
    # Actualizar la visualización
    fig.canvas.draw_idle()
    print("Visualización actualizada")

# Parámetros de la simulación
Uinf = 60            # Velocidad del flujo libre
R = 1                # Radio de referencia
dx = 0.05            # Paso en X (resolución moderada para velocidad)
dy = 0.05            # Paso en Y
Xrange = 4 * R       # Rango en X
Yrange = 3 * R       # Rango en Y
  
# Número de iteraciones para la solución numérica (reducido para interactividad)
iterations = 200
    
# Resolución de la malla
Xdiv = int(Xrange / dx + 1)
Ydiv = int(Yrange / dy + 1)
 
print("Configurando simulación...")
print(f"Speed = {Uinf}")
print(f"Domain dimensions: {Xrange}x{Yrange}")
print(f"Resolution: dx={dx}, dy={dy}")
print("--------------------")
    
# Crear la malla
x, y, X, Y = create_mesh(Xrange, Yrange, Xdiv, Ydiv)
  
# Por defecto, empezar con el perfil aerodinámico
Xpoints, Ypoints = Airfoil_model(R, Xrange, Yrange, alpha=0)
model, Xaranged, Yaranged = create_model(Xpoints, Ypoints, dx, dy, Xdiv, Ydiv, x, y)
   
# Valor inicial de C (non-lifting)
C_default = Uinf * Yrange * 0.5
    
# Configurar la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.3)  # Ajustar espacio para los controles
  
# Crear un eje para el deslizador de C
ax_slider = plt.axes([0.25, 0.20, 0.65, 0.03])
  
# Crear un deslizador para C
slider = Slider(ax_slider, 'C', 0.0, Uinf*Yrange, valinit=Uinf*Yrange*0.5, valstep=0.5)
    
# Crear un eje para el deslizador de ángulo
ax_angle_slider = plt.axes([0.25, 0.15, 0.65, 0.03])
    
# Crear un deslizador para el ángulo de ataque
angle_slider = Slider(ax_angle_slider, 'Ángulo de Ataque (°)', -15, 15, valinit=0, valstep=1)

# Crear un eje para el deslizador de dx&dy
ax_res_slider = plt.axes([0.25, 0.1, 0.65, 0.03])

# Crear un deslizador para dx&dy
res_slider = Slider(ax_res_slider, "Resolución de malla", 0.001, 0.1, valinit=0.05, valstep=0.001)
   
# Radio Buttons para seleccionar modelo
ax_modos = plt.axes([0.25, 0, 0.65, 0.10])
modos = RadioButtons(ax_modos, ('Perfil Aerodinámico', 'Cilindro', "Otro Modelo"), active=0)
    
# Texto para interpretación de C
ax_info = plt.axes([0.25, 0.01, 0.65, 0.02])
info_text = plt.text(0.5, 0.5, 'C = 0.5 (Sin sustentación)', transform=ax_info.transAxes, ha='center', va='center')
ax_info.axis('off')
    
# Texto de estado
ax_status = plt.axes([0.25, 0.005, 0.65, 0.02])
status_text = plt.text(0.5, 0.5, 'Listo para calcular.', transform=ax_status.transAxes, ha='center', va='center')
ax_status.axis('off')
    
# Almacenar todos los datos necesarios en un diccionario para pasar a las funciones de actualización
simulation_data = {
    'Uinf': Uinf,
    'Yrange': Yrange,
    'Xdiv': Xdiv,
    'Ydiv': Ydiv,
    'x': x,
    'y': y,
    'X': X,
    'Y': Y,
    'model': model,
    'Xaranged': Xaranged,
    'Yaranged': Yaranged,
    'Xplot': Xpoints,  # Coordenadas de alta definición para dibujar
    'Yplot': Ypoints,
    'ax': ax,
    'fig': fig,
    'slider': slider,
    'angle_slider': angle_slider,
    "res_slider" : res_slider,
    'status_text': status_text,
    'iterations': iterations,
    'info_text': info_text,
    'modos': modos,
    'dx': dx,
    'dy': dy,
    'R': R,
    'Xrange': Xrange,
    'cbar': None,
    'current_model': 'Perfil Aerodinámico'
}
    
# Conectar la función de actualización a los sliders
slider.on_changed(lambda val: update(val, simulation_data))
angle_slider.on_changed(lambda val: update(val, simulation_data))
res_slider.on_changed(lambda val: update(val, simulation_data))
    
# Conectar la función de cambio de modelo al modos button
modos.on_clicked(lambda label: change_model(label, simulation_data))
    
# Realizar la visualización inicial
update(None, simulation_data)
    
print("Simulación lista. Ajusta el modelo o los sliders para ver cambios automáticos.")
plt.show()
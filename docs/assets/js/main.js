// Import Three.js from UNPKG (same as the test that worked)
import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';
import { PLYLoader } from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/PLYLoader.js';
// Global variables for Three.js
let scenes = {};
let cameras = {};
let renderers = {};
let controls = {};

// Initialize 3D viewer
function init3DViewer(containerId) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container ${containerId} not found`);
        return;
    }
    
    const width = container.offsetWidth;
    const height = container.offsetHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    scenes[containerId] = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 0, 3);
    cameras[containerId] = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);
    renderers[containerId] = renderer;

    // Controls
    const control = new OrbitControls(camera, renderer.domElement);
    control.enableDamping = true;
    control.dampingFactor = 0.05;
    control.screenSpacePanning = false;
    control.minDistance = 1;
    control.maxDistance = 10;
    controls[containerId] = control;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(1, 1, 1);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
    directionalLight2.position.set(-1, -1, -1);
    scene.add(directionalLight2);

    // Add a point light for better face illumination
    const pointLight = new THREE.PointLight(0xffffff, 0.3);
    pointLight.position.set(0, 0, 2);
    scene.add(pointLight);

    // Start animation
    animate(containerId);
}

// Resolve model path
function resolveModelPath(p) {
    // Allow absolute URLs or leading slash; otherwise read from assets/models/
    if (p.startsWith('http://') || p.startsWith('https://')) return p;
    
    // For GitHub Pages, we need relative paths
    const basePath = window.location.pathname.includes('.html') 
        ? window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/'))
        : window.location.pathname.replace(/\/$/, '');
    
    // If running on GitHub Pages (not localhost)
    if (window.location.hostname.includes('github.io')) {
        if (p.startsWith('/')) {
            // Absolute path from repo root
            return `${basePath}${p}`;
        }
        // Relative path
        return `${basePath}/assets/models/${p}`;
    }
    
    // For local development
    if (p.startsWith('/')) return p;
    return `assets/models/${p}`;
}

// Add loading spinner
function addSpinner(container) {
    let spinner = container.querySelector('.loading-spinner');
    if (!spinner) {
        spinner = document.createElement('div');
        spinner.className = 'loading-spinner';
        spinner.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading model...';
        spinner.style.position = 'absolute';
        spinner.style.left = '50%';
        spinner.style.top = '50%';
        spinner.style.transform = 'translate(-50%, -50%)';
        spinner.style.color = '#007bff';
        spinner.style.fontSize = '1.2rem';
        spinner.style.background = 'rgba(255,255,255,0.9)';
        spinner.style.padding = '20px';
        spinner.style.borderRadius = '8px';
        spinner.style.boxShadow = '0 2px 10px rgba(0,0,0,0.1)';
        container.appendChild(spinner);
    }
    return spinner;
}

// Load model (GLB or PLY)
export function loadModel(modelPath, containerId) {
    const scene = scenes[containerId];
    if (!scene) {
        console.error(`Scene for ${containerId} not initialized`);
        return;
    }
    
    const container = document.getElementById(containerId);
    const spinner = addSpinner(container);

    // Remove previous model if any
    const oldModel = scene.getObjectByName('model');
    if (oldModel) {
        scene.remove(oldModel);
    }

    const fullPath = resolveModelPath(modelPath);
    const extension = modelPath.split('.').pop().toLowerCase();

    let loader;
    if (extension === 'glb' || extension === 'gltf') {
        loader = new GLTFLoader();
    } else if (extension === 'ply') {
        loader = new PLYLoader();
    } else {
        console.error(`Unsupported file format: ${extension}`);
        spinner.textContent = 'Unsupported file format';
        return;
    }

    // Load based on file type
    if (extension === 'glb' || extension === 'gltf') {
        loader.load(
            fullPath,
            (gltf) => {
                spinner.remove();
                const model = gltf.scene;
                model.name = 'model';
                
                // Enable shadows
                model.traverse((child) => {
                    if (child.isMesh) {
                        child.castShadow = true;
                        child.receiveShadow = true;
                    }
                });
                
                // Center and scale model
                centerAndScaleModel(model);
                scene.add(model);
            },
            (progress) => {
                const percent = (progress.loaded / progress.total * 100).toFixed(0);
                spinner.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Loading... ${percent}%`;
            },
            (error) => {
                console.error('GLB load error:', error);
                spinner.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Error loading model';
            }
        );
    } else if (extension === 'ply') {
        loader.load(
            fullPath,
            (geometry) => {
                spinner.remove();
                
                // Create material for PLY
                const material = new THREE.MeshPhongMaterial({
                    color: 0xaaaaaa,
                    specular: 0x111111,
                    shininess: 200,
                    vertexColors: geometry.hasAttribute('color')
                });
                
                const mesh = new THREE.Mesh(geometry, material);
                mesh.name = 'model';
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                
                // Center and scale model
                centerAndScaleModel(mesh);
                scene.add(mesh);
            },
            (progress) => {
                const percent = (progress.loaded / progress.total * 100).toFixed(0);
                spinner.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Loading... ${percent}%`;
            },
            (error) => {
                console.error('PLY load error:', error);
                spinner.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Error loading model';
            }
        );
    }
}

// Center and scale model to fit view
function centerAndScaleModel(model) {
    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    
    // Center the model
    model.position.x = -center.x;
    model.position.y = -center.y;
    model.position.z = -center.z;
    
    // Scale to fit
    const maxDim = Math.max(size.x, size.y, size.z);
    const scale = 2 / maxDim;
    model.scale.setScalar(scale);
}

// Animation loop
function animate(containerId) {
    requestAnimationFrame(() => animate(containerId));
    
    if (controls[containerId]) {
        controls[containerId].update();
    }
    
    if (renderers[containerId] && scenes[containerId] && cameras[containerId]) {
        renderers[containerId].render(scenes[containerId], cameras[containerId]);
    }
}

// Handle window resize
window.addEventListener('resize', () => {
    Object.keys(renderers).forEach(containerId => {
        const container = document.getElementById(containerId);
        if (container) {
            const width = container.offsetWidth;
            const height = container.offsetHeight;
            
            if (cameras[containerId]) {
                cameras[containerId].aspect = width / height;
                cameras[containerId].updateProjectionMatrix();
            }
            
            if (renderers[containerId]) {
                renderers[containerId].setSize(width, height);
            }
        }
    });
});

// Initialize viewers when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing 3D viewers...');
    
    // Initialize all model viewers
    const viewers = document.querySelectorAll('.model-viewer');
    viewers.forEach(viewer => {
        if (viewer.id && viewer.id !== 'input-viewer') {
            console.log(`Initializing viewer: ${viewer.id}`);
            init3DViewer(viewer.id);
            
            // Load a default cube if no model is available
            addDefaultCube(viewer.id);
        }
    });

    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
});

// Add a default cube as placeholder
function addDefaultCube(containerId) {
    const scene = scenes[containerId];
    if (!scene) return;
    
    // Create a simple cube as default
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshPhongMaterial({ 
        color: 0x007bff,
        specular: 0x111111,
        shininess: 200
    });
    const cube = new THREE.Mesh(geometry, material);
    cube.name = 'model';
    cube.castShadow = true;
    cube.receiveShadow = true;
    scene.add(cube);
    
    // Add text below cube
    const container = document.getElementById(containerId);
    const info = document.createElement('div');
    info.style.position = 'absolute';
    info.style.bottom = '10px';
    info.style.left = '50%';
    info.style.transform = 'translateX(-50%)';
    info.style.color = '#666';
    info.style.fontSize = '0.9rem';
    info.textContent = 'Default cube - Click buttons to load models';
    container.appendChild(info);
}

// Make loadModel globally accessible
window.loadModel = loadModel;

// Log to confirm it's loaded
console.log('main.js loaded, loadModel available:', typeof window.loadModel);
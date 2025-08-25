import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';

// Global variables for Three.js
let scenes = {};
let cameras = {};
let renderers = {};
let controls = {};

// Initialize 3D viewer
function init3DViewer(containerId) {
    const container = document.getElementById(containerId);
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
    container.appendChild(renderer.domElement);
    renderers[containerId] = renderer;

    // Controls
    const control = new THREE.OrbitControls(camera, renderer.domElement);
    control.enableDamping = true;
    control.dampingFactor = 0.05;
    controls[containerId] = control;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
    directionalLight2.position.set(-1, -1, -1);
    scene.add(directionalLight2);

    // Start animation
    animate(containerId);
}

// Load GLB model
function resolveModelPath(p) {
  // allow absolute URLs or leading slash; otherwise read from assets/models/
  if (p.startsWith('http://') || p.startsWith('https://') || p.startsWith('/')) return p;
  return `assets/models/${p}`;
}

function addSpinner(container) {
  let s = container.querySelector('.loading-spinner');
  if (!s) {
    s = document.createElement('div');
    s.className = 'loading-spinner';
    s.textContent = 'Loading...';
    s.style.position = 'absolute';
    s.style.left = '50%';
    s.style.top  = '50%';
    s.style.transform = 'translate(-50%, -50%)';
    container.appendChild(s);
  }
  return s;
}

export function loadModel(modelPath, containerId) {
  const scene = scenes[containerId];
  const container = document.getElementById(containerId);
  const spinner = addSpinner(container);

  // remove previous model if any
  const old = scene.getObjectByName('model');
  if (old) scene.remove(old);

  const loader = new GLTFLoader();
  loader.load(
    resolveModelPath(modelPath),
    (gltf) => {
      spinner.remove();
      const model = gltf.scene;
      model.name = 'model';
      // center & scale to a reasonable size
      const box = new THREE.Box3().setFromObject(model);
      const center = box.getCenter(new THREE.Vector3());
      model.position.sub(center);

      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z) || 1.0;
      model.scale.setScalar(2 / maxDim);

      scene.add(model);
    },
    undefined,
    (err) => {
      console.error('GLB load error:', err);
      spinner.textContent = 'Error loading model';
    }
  );
}

// Animation loop
function animate(containerId) {
    requestAnimationFrame(() => animate(containerId));
    
    controls[containerId].update();
    renderers[containerId].render(scenes[containerId], cameras[containerId]);
}

// Handle window resize
window.addEventListener('resize', () => {
    Object.keys(renderers).forEach(containerId => {
        const container = document.getElementById(containerId);
        const width = container.offsetWidth;
        const height = container.offsetHeight;
        
        cameras[containerId].aspect = width / height;
        cameras[containerId].updateProjectionMatrix();
        renderers[containerId].setSize(width, height);
    });
});

// Initialize viewers when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Initialize all model viewers
    const viewers = document.querySelectorAll('.model-viewer');
    viewers.forEach(viewer => {
        if (viewer.id && viewer.id !== 'input-viewer') {
            init3DViewer(viewer.id);
            
            // Load default model
            loadModel('default.ply', viewer.id);
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

// Add loading spinner styles
const style = document.createElement('style');
style.textContent = `
    .loading-spinner {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 1.2rem;
        color: #666;
        background: rgba(255,255,255,0.9);
        padding: 20px;
        border-radius: 8px;
    }
`;
document.head.appendChild(style);
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Upload, 
  Image as ImageIcon, 
  Zap, 
  RefreshCcw, 
  Layers, 
  Download, 
  CheckCircle2, 
  ChevronRight,
  Info,
  AlertCircle,
  Eye,
  EyeOff
} from 'lucide-react';
import { analyzeOffroadImage, SegmentationResult } from './lib/gemini';
import Dither from './components/Dither';

// Classes for segmentation
const SEGMENTATION_CLASSES: Record<string, { color: string; label: string; hex: string; detectedColor: string }> = {
  sky: { color: 'rgba(34, 211, 238, 0.35)', label: 'Sky', hex: '#22d3ee', detectedColor: 'rgba(34, 211, 238, 0.65)' },
  ground: { color: 'rgba(251, 191, 36, 0.35)', label: 'Ground', hex: '#fbbf24', detectedColor: 'rgba(251, 191, 36, 0.65)' },
  road: { color: 'rgba(129, 140, 153, 0.35)', label: 'Road', hex: '#818c99', detectedColor: 'rgba(129, 140, 153, 0.65)' },
  gravel: { color: 'rgba(168, 162, 158, 0.35)', label: 'Gravel', hex: '#a8a29e', detectedColor: 'rgba(168, 162, 158, 0.65)' },
  vegetation: { color: 'rgba(34, 197, 94, 0.35)', label: 'Vegetation', hex: '#22c55e', detectedColor: 'rgba(34, 197, 94, 0.65)' },
  trees: { color: 'rgba(21, 128, 61, 0.35)', label: 'Trees', hex: '#15803d', detectedColor: 'rgba(21, 128, 61, 0.65)' },
  rocks: { color: 'rgba(100, 116, 139, 0.35)', label: 'Rocks', hex: '#64748b', detectedColor: 'rgba(100, 116, 139, 0.65)' },
  water: { color: 'rgba(59, 130, 246, 0.35)', label: 'Water', hex: '#3b82f6', detectedColor: 'rgba(59, 130, 246, 0.65)' },
  obstacle: { color: 'rgba(239, 68, 68, 0.35)', label: 'Obstacle', hex: '#ef4444', detectedColor: 'rgba(239, 68, 68, 0.65)' },
  human: { color: 'rgba(168, 85, 247, 0.35)', label: 'Human', hex: '#a855f7', detectedColor: 'rgba(168, 85, 247, 0.65)' },
  building: { color: 'rgba(75, 85, 99, 0.35)', label: 'Building', hex: '#4b5563', detectedColor: 'rgba(75, 85, 99, 0.65)' },
};

function SegmentationCanvas({ 
  regions, 
  showOverlay, 
  containerWidth, 
  containerHeight,
  detectedClasses
}: { 
  regions: SegmentationResult['regions']; 
  showOverlay: boolean;
  containerWidth: number;
  containerHeight: number;
  detectedClasses: string[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (showOverlay) {
      regions.forEach(region => {
        const classInfo = SEGMENTATION_CLASSES[region.class];
        if (!classInfo) return;

        const isHighlighted = detectedClasses.includes(region.class);
        ctx.fillStyle = isHighlighted ? classInfo.detectedColor : classInfo.color;
        ctx.strokeStyle = classInfo.hex;
        ctx.lineWidth = isHighlighted ? 2.5 : 1.5;
        ctx.lineJoin = 'round';

        ctx.beginPath();
        region.points.forEach((point, index) => {
          const x = (point[0] / 1000) * canvas.width;
          const y = (point[1] / 1000) * canvas.height;
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      });
    }
  }, [regions, showOverlay, containerWidth, containerHeight, detectedClasses]);

  return (
    <canvas 
      ref={canvasRef} 
      width={containerWidth} 
      height={containerHeight}
      className="absolute inset-0 pointer-events-none transition-opacity duration-500 segment-overlay"
      style={{ opacity: showOverlay ? 0.75 : 0 }}
    />
  );
}

export default function App() {
  const [image, setImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<SegmentationResult | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageContainerRef = useRef<HTMLDivElement>(null);

  // Handle container resize
  useEffect(() => {
    const observer = new ResizeObserver(entries => {
      if (entries[0]) {
        const { width, height } = entries[0].contentRect;
        setDimensions({ width, height });
      }
    });

    if (imageContainerRef.current) {
      observer.observe(imageContainerRef.current);
    }

    return () => observer.disconnect();
  }, [image]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please upload a valid image file (JPG or PNG).');
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        setImage(reader.result as string);
        setResult(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const runSegmentation = async () => {
    if (!image) return;
    setIsProcessing(true);
    setError(null);
    
    try {
      const segmentation = await analyzeOffroadImage(image);
      setResult(segmentation);
    } catch (err) {
      setError('An error occurred during segmentation. Please try again.');
      console.error(err);
    } finally {
      setIsProcessing(false);
    }
  };

  const reset = () => {
    setImage(null);
    setResult(null);
    setError(null);
  };

  const detectedClasses = result?.regions.map(r => r.class) || [];
  const detectedUniqueClasses = Array.from(new Set(detectedClasses));

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-indigo-500/30 relative overflow-x-hidden">
      {/* Background Effect */}
      <div className="fixed inset-0 z-0 pointer-events-none opacity-50">
        <Dither
          waveColor={[0.3, 0.2, 0.8]}
          disableAnimation={false}
          enableMouseInteraction
          mouseRadius={0.5}
          colorNum={4}
          pixelSize={2}
          waveAmplitude={0.12}
          waveFrequency={2.5}
          waveSpeed={0.04}
        />
      </div>

      <div className="relative z-10">
        {/* Header */}
        <header className="border-b border-white/5 bg-slate-950/40 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-4 cursor-pointer" onClick={() => window.location.reload()}>
            <div className="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-900/20 transition-transform active:scale-95">
              <Zap className="text-white w-6 h-6 fill-current" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-bold text-xl tracking-tight text-white">TerraScan AI</h1>
              </div>
              <p className="text-slate-400 text-xs font-medium">Advanced terrain analysis and scene understanding.</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
             <button 
              onClick={reset}
              className="px-4 py-2 text-sm font-medium text-slate-300 bg-white/5 border border-white/10 rounded-lg hover:bg-white/10 transition-colors shadow-sm"
            >
              Reset Session
            </button>
            <button className="px-4 py-2 text-sm font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 transition-colors shadow-sm shadow-indigo-900/20">
              Download Results
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-12">
        <div className="grid lg:grid-cols-12 gap-8 items-start">
          {/* Main Interaction Area */}
          <div className="lg:col-span-12 space-y-8">
            <AnimatePresence mode="wait">
              {!image ? (
                <motion.div
                  key="upload"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="relative group"
                >
                  <div 
                    onClick={() => fileInputRef.current?.click()}
                    className="border-2 border-dashed border-white/10 rounded-[2rem] p-12 flex flex-col items-center justify-center gap-8 bg-slate-900/50 hover:border-indigo-500/50 hover:bg-indigo-500/5 transition-all cursor-pointer min-h-[500px] shadow-sm hover:shadow-2xl hover:shadow-indigo-500/10 group"
                  >
                    <div className="w-24 h-24 rounded-[2rem] bg-slate-900 flex items-center justify-center text-slate-500 group-hover:bg-indigo-600 group-hover:text-white group-hover:rotate-6 transition-all duration-500 shadow-inner border border-white/5">
                      <Upload className="w-10 h-10" />
                    </div>
                    <div className="text-center">
                      <h2 className="text-3xl font-bold text-white tracking-tight">Drop terrain image here</h2>
                      <p className="text-slate-400 mt-3 max-w-md mx-auto leading-relaxed font-medium capitalize">
                        Supports high-resolution JPG and PNG. Our model is optimized for desert, rocky, and unstructured environments.
                      </p>
                    </div>
                    <div className="px-8 py-4 bg-indigo-600 text-white rounded-2xl font-bold text-sm tracking-wide shadow-xl shadow-indigo-900/20 group-hover:bg-indigo-700 transition-all active:scale-95">
                      Select Laboratory Data
                    </div>
                  </div>
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    onChange={handleFileUpload} 
                    className="hidden" 
                    accept="image/*"
                  />
                </motion.div>
              ) : (
                <motion.div
                  key="preview"
                  initial={{ opacity: 0, scale: 0.98 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="space-y-8"
                >
                  <div className="grid md:grid-cols-2 gap-8 items-center">
                    {/* Input Side */}
                    <div className="space-y-4">
                      <div className="flex justify-between items-center px-1">
                        <span className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Input Stream</span>
                        <span className="text-xs text-slate-500 font-mono">LAB_STREAM_01</span>
                      </div>
                      <div className="relative aspect-[4/3] rounded-[2rem] overflow-hidden shadow-2xl border border-white/10 bg-slate-900">
                        <img 
                          src={image} 
                          alt="Origin" 
                          className="w-full h-full object-cover"
                        />
                      </div>
                    </div>

                    {/* Output Side */}
                    <div className="space-y-4">
                      <div className="flex justify-between items-center px-1">
                        <span className="text-xs font-semibold text-indigo-400 uppercase tracking-widest">AI Prediction</span>
                        {result && (
                          <div className="flex items-center gap-2">
                            <span className="flex h-2 w-2 rounded-full bg-green-500 animate-pulse"></span>
                            <span className="text-xs text-slate-500 font-medium tracking-tight">Inference Complete</span>
                          </div>
                        )}
                      </div>
                      <div 
                        ref={imageContainerRef}
                        className="relative aspect-[4/3] rounded-[2rem] overflow-hidden shadow-2xl border border-indigo-900/50 bg-slate-900 group"
                      >
                        <img 
                          src={image} 
                          alt="Segmented" 
                          className="w-full h-full object-cover opacity-100"
                        />
                        
                        {/* Segmentation Overlay */}
                        {result && (
                          <SegmentationCanvas 
                            regions={result.regions} 
                            showOverlay={showOverlay}
                            containerWidth={dimensions.width}
                            containerHeight={dimensions.height}
                            detectedClasses={detectedClasses}
                          />
                        )}

                        {result && (
                          <div className="absolute top-4 right-4 glass-panel px-4 py-2 rounded-full shadow-lg flex flex-col gap-0.5 z-10 font-display">
                            <span className="text-[10px] font-bold text-white tracking-tighter uppercase whitespace-nowrap">Confidence: {(result.confidence * 100).toFixed(1)}%</span>
                            <span className="text-[10px] font-bold text-indigo-400 tracking-tighter uppercase whitespace-nowrap">mIoU: {result.mIoU.toFixed(3)}</span>
                          </div>
                        )}

                        {isProcessing && (
                          <div className="absolute inset-0 bg-slate-950/60 backdrop-blur-xl flex flex-col items-center justify-center gap-6 z-20 text-white">
                            <div className="relative">
                              <div className="w-20 h-20 border-4 border-indigo-500/10 rounded-full" />
                              <div className="w-20 h-20 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin absolute inset-0" />
                            </div>
                            <div className="text-center">
                              <p className="font-bold text-white text-lg">Neural Graph Analysis</p>
                              <p className="text-[10px] text-slate-400 font-bold tracking-widest uppercase mt-1 px-4 py-1 bg-white/5 rounded-full inline-block">Pixel-Level Classification</p>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Actions & Results Section */}
                  <div className="glass-panel rounded-[2rem] p-8 shadow-sm border border-white/5">
                    <div className="flex flex-col md:flex-row items-center justify-between gap-10">
                      <div className="flex items-center gap-10">
                        <div className="space-y-3">
                          <span className="text-[10px] uppercase font-bold text-slate-500 tracking-tight">Terrain Legend</span>
                          <div className="flex flex-wrap gap-5">
                            {Object.entries(SEGMENTATION_CLASSES).map(([key, data]) => {
                              const isDetected = detectedClasses.includes(key);
                              return (
                                <div 
                                  key={key} 
                                  className={`flex items-center gap-2 transition-all duration-500 ${
                                    result && !isDetected ? 'opacity-30' : 'opacity-100'
                                  }`}
                                >
                                  <div className="w-3 h-3 rounded-[3px] shadow-sm" style={{ backgroundColor: data.hex }} />
                                  <span className="text-[11px] font-bold text-slate-400 tracking-tight">{data.label}</span>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>

                      <div className="hidden md:block h-16 w-[1px] bg-white/10"></div>

                      <div className="flex items-center gap-8">
                        <div className="text-center space-y-1">
                          <div className="text-xl font-bold text-white">1,024<span className="text-[10px] text-slate-500 ml-0.5 uppercase tracking-tighter">px</span></div>
                          <div className="text-[10px] text-slate-500 uppercase font-bold tracking-tighter">Resolution</div>
                        </div>
                        <div className="text-center space-y-1">
                          <div className="text-xl font-bold text-indigo-400">PyTorch<span className="text-[10px] text-slate-500 ml-0.5 font-mono">1.12</span></div>
                          <div className="text-[10px] text-slate-500 uppercase font-bold tracking-tighter">Engine</div>
                        </div>
                        
                        <div className="flex items-center gap-2 bg-white/5 p-1.5 rounded-2xl border border-white/10">
                          <button 
                            disabled={isProcessing}
                            onClick={!result ? runSegmentation : () => setShowOverlay(!showOverlay)}
                            className={`px-5 py-2.5 rounded-xl text-xs font-bold transition-all shadow-sm ${
                              result 
                                ? (showOverlay ? 'bg-indigo-600 text-white' : 'bg-white/10 text-slate-300 hover:bg-white/20')
                                : 'bg-indigo-600 text-white'
                            }`}
                          >
                            {!result ? 'Start Prediction' : (showOverlay ? 'Hide Prediction' : 'Show Prediction')}
                          </button>
                        </div>
                      </div>
                    </div>

                    {result && !isProcessing && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        className="mt-8 pt-8 border-t border-white/10"
                      >
                         <h4 className="text-[10px] uppercase font-bold text-slate-500 tracking-tight mb-4 flex items-center gap-2">
                           <CheckCircle2 className="w-3 h-3 text-green-500" />
                           Laboratory Analysis Summary
                         </h4>
                         <p className="text-lg font-medium text-slate-300 leading-relaxed italic pr-12">
                           "{result.summary}"
                         </p>

                         <p className="mt-6 text-sm text-slate-400">
                           Detected terrain colors from the uploaded image. Each card shows the terrain type and its representative color.
                         </p>

                         <p className="mt-2 text-sm text-slate-300">
                           Detected colors: {detectedUniqueClasses.map((classKey) => `${SEGMENTATION_CLASSES[classKey]?.label ?? classKey} (${SEGMENTATION_CLASSES[classKey]?.hex ?? 'N/A'})`).join(', ')}.
                         </p>

                         <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                           {detectedUniqueClasses.map((classKey) => {
                             const classInfo = SEGMENTATION_CLASSES[classKey];
                             if (!classInfo) return null;
                             return (
                               <div key={classKey} className="border border-white/10 rounded-3xl bg-slate-900/80 p-4 shadow-sm">
                                 <div className="flex items-center gap-3">
                                   <div className="w-8 h-8 rounded-xl" style={{ backgroundColor: classInfo.hex, boxShadow: `0 0 0 1px ${classInfo.hex}33` }} />
                                   <div>
                                     <p className="text-sm font-bold text-white">{classInfo.label}</p>
                                     <p className="text-[11px] text-slate-400 uppercase tracking-widest">{classInfo.hex}</p>
                                   </div>
                                 </div>
                               </div>
                             );
                           })}
                         </div>
                      </motion.div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {error && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-6 bg-red-950/20 border border-red-500/20 rounded-3xl flex items-center gap-4 text-red-400 shadow-sm"
              >
                <AlertCircle className="w-6 h-6 flex-shrink-0" />
                <p className="text-sm font-bold">{error}</p>
              </motion.div>
            )}
          </div>
        </div>
      </main>

      <footer className="max-w-6xl mx-auto px-6 py-12 flex flex-col md:flex-row justify-between items-center gap-6 opacity-60">
        <div className="flex items-center gap-4 text-slate-400">
          <span className="text-[10px] font-bold uppercase tracking-widest">Model Status:</span>
          <span className="text-[10px] font-bold text-green-400 bg-green-500/10 px-3 py-1 rounded-full border border-green-500/20">Neural Network Online</span>
        </div>
        <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest text-center">
          Designed for Autonomous Navigation Research & Academic Evaluation
        </p>
      </footer>
      </div>
    </div>
  );
}

import { ArrowRight, BarChart3, Layers, ShieldCheck, Upload, Waves } from "lucide-react";
import { Link } from "react-router-dom";
import { Logo, ThemeToggle } from "../components/Layout";
import { Disclaimer } from "../components/ui";

/** Procedural illustration (not patient data and not a model output). */
function PhantomHero() {
  return (
    <figure className="relative">
      <svg viewBox="0 0 400 400" className="w-full max-w-[460px] mx-auto rounded-2xl bg-[#05080f] border border-line shadow-card" role="img" aria-label="Illustrative synthetic axial brain phantom with colour-coded tumor sub-regions">
        <defs>
          <radialGradient id="tissue" cx="50%" cy="50%" r="55%"><stop offset="0" stopColor="#8a97a8" /><stop offset="0.7" stopColor="#5d6b7d" /><stop offset="1" stopColor="#2c3644" /></radialGradient>
          <filter id="soft"><feGaussianBlur stdDeviation="1.2" /></filter>
        </defs>
        <ellipse cx="200" cy="205" rx="150" ry="172" fill="none" stroke="#c8d0dc" strokeWidth="9" opacity="0.5" />
        <ellipse cx="200" cy="205" rx="138" ry="160" fill="url(#tissue)" />
        <path d="M200 60 C190 130 210 150 200 205 C190 260 210 280 200 350" stroke="#1d2530" strokeWidth="5" fill="none" opacity="0.8" />
        <ellipse cx="170" cy="190" rx="22" ry="40" fill="#10151d" />
        <ellipse cx="230" cy="190" rx="22" ry="40" fill="#10151d" />
        <g filter="url(#soft)">
          <ellipse cx="262" cy="150" rx="52" ry="46" fill="#f5c542" opacity="0.55" />
          <ellipse cx="262" cy="150" rx="34" ry="30" fill="#ef4444" opacity="0.6" />
          <ellipse cx="262" cy="150" rx="16" ry="14" fill="#3b82f6" opacity="0.75" />
        </g>
        <ellipse cx="262" cy="150" rx="52" ry="46" fill="none" stroke="#f5c542" strokeWidth="1.5" strokeDasharray="4 3" />
      </svg>
      <figcaption className="text-center text-xs text-mute mt-3">Illustrative phantom — not patient data, not a model prediction.</figcaption>
    </figure>
  );
}

const FEATURES = [
  { icon: Layers, title: "Multi-modal U-Net segmentation", text: "Four-channel (T1, T1ce, T2, FLAIR) 2D U-Net and Attention U-Net producing necrotic core, edema and enhancing tumor masks." },
  { icon: ShieldCheck, title: "Confidence you can interrogate", text: "Voxel, region and scan-level scores with TTA and Monte-Carlo dropout. Calibrated on validation data only, never presented as certainty." },
  { icon: Waves, title: "Uncertainty maps", text: "Predictive entropy and mutual-information heatmaps, linked to observed error when ground truth exists (AUROC)." },
  { icon: BarChart3, title: "Honest evaluation", text: "Patient-level Dice, IoU, HD95 in mm, ECE and Brier. Experiments are shown as planned until they have actually been run." },
];
const STEPS = ["Upload four co-registered NIfTI volumes", "Validate grid, spacing and orientation", "Brain-masked z-score normalisation", "U-Net inference (optional TTA / MC-dropout)", "Inspect slices, volumes, confidence and uncertainty", "Export mask, preview and metrics"];

export default function Landing() {
  return (
    <div className="min-h-full">
      <header className="h-16 px-5 sm:px-10 flex items-center justify-between border-b border-line bg-surface/70 backdrop-blur sticky top-0 z-20">
        <Logo />
        <nav className="flex items-center gap-2 text-sm" aria-label="Site">
          <a href="#features" className="btn-ghost hidden sm:inline-flex">Features</a>
          <a href="#workflow" className="btn-ghost hidden sm:inline-flex">Workflow</a>
          <ThemeToggle />
          <Link to="/app" className="btn-primary btn-sm">Launch workspace</Link>
        </nav>
      </header>

      <section className="px-5 sm:px-10 py-14 lg:py-20 max-w-[1250px] mx-auto grid lg:grid-cols-[1.1fr_1fr] gap-12 items-center">
        <div>
          <span className="inline-flex items-center gap-2 text-xs font-medium text-ai bg-ai/10 rounded-full px-3 py-1">Research platform · BraTS-style multi-modal MRI</span>
          <h1 className="text-4xl sm:text-5xl font-semibold tracking-tight mt-5 leading-[1.08]">Segment glioma sub-regions.<br /><span className="text-brand">Measure how far to trust the result.</span></h1>
          <p className="text-lg text-mute mt-5 max-w-xl">NeuroVista AI pairs a trainable U-Net pipeline with an interactive multi-planar viewer, calibrated confidence and uncertainty analysis — built for reproducible research, with every number traceable to the model that produced it.</p>
          <div className="flex flex-wrap gap-3 mt-8">
            <Link to="/app" className="btn-primary h-11 px-6">Launch workspace <ArrowRight className="h-4 w-4" /></Link>
            <Link to="/app/upload" className="btn-secondary h-11 px-6"><Upload className="h-4 w-4" />Upload a scan</Link>
          </div>
        </div>
        <PhantomHero />
      </section>

      <section id="features" className="px-5 sm:px-10 py-14 bg-surface border-y border-line">
        <div className="max-w-[1250px] mx-auto">
          <h2 className="text-2xl font-semibold tracking-tight">What the platform does</h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5 mt-8">
            {FEATURES.map((f) => (
              <div key={f.title} className="card p-5">
                <f.icon className="h-5 w-5 text-brand" />
                <h3 className="font-semibold mt-3">{f.title}</h3>
                <p className="text-sm text-mute mt-1.5">{f.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="workflow" className="px-5 sm:px-10 py-14 max-w-[1250px] mx-auto">
        <h2 className="text-2xl font-semibold tracking-tight">Scientific workflow</h2>
        <ol className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 mt-8">
          {STEPS.map((s, i) => (
            <li key={s} className="flex gap-4 items-start card p-4">
              <span className="num grid place-items-center h-8 w-8 rounded-full bg-navy text-brand text-sm font-semibold shrink-0">{i + 1}</span>
              <span className="text-sm pt-1.5">{s}</span>
            </li>
          ))}
        </ol>
      </section>

      <footer className="px-5 sm:px-10 py-8 border-t border-line bg-surface">
        <div className="max-w-[1250px] mx-auto space-y-3">
          <Disclaimer />
          <p className="text-xs text-mute">Not a medical device. Datasets such as BraTS carry their own licences; see DATASET_GUIDE.md.</p>
        </div>
      </footer>
    </div>
  );
}

'use client';

import { useState } from 'react';

export default function LiveDemo() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/predict-disease', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error fetching prediction:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section id="demo" className="max-w-[1100px] mx-auto px-6 py-24">
      {/* Section Label */}
      <div className="flex items-center gap-3 mb-4">
        <div className="w-6 h-[1px] bg-accent-lime"></div>
        <span className="text-[11px] font-medium text-accent-lime uppercase tracking-[0.15em]">Live Demo</span>
      </div>
      <h2 className="font-syne font-extrabold text-5xl text-white leading-tight mb-12">
        Type symptoms.<br />Get answers.
      </h2>

      <div className="grid md:grid-cols-2 gap-6 mb-12">
        {/* Left Card: Input */}
        <div className="relative bg-card border border-accent-lime/20 rounded-[20px] p-8 overflow-hidden group">
          <div className="absolute top-[-50px] right-[-50px] w-[150px] h-[150px] bg-accent-lime/10 rounded-full blur-[40px] group-hover:bg-accent-lime/15 transition-all"></div>
          
          <span className="block text-[12px] font-medium text-muted uppercase mb-4">Symptom Description</span>
          <form onSubmit={handleSubmit}>
            <textarea
              className="w-full h-[120px] bg-background/50 border border-accent-teal/30 rounded-xl p-4 font-dm-sans text-sm text-white placeholder:text-muted focus:border-accent-lime/50 focus:outline-none transition-all mb-4"
              placeholder="e.g. High fever, dry cough, and loss of taste..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            
            <div className="flex flex-wrap gap-2 mb-6">
              {["Rheumatoid", "Diabetes", "Cardiac", "Lupus"].map(tag => (
                <button 
                  key={tag}
                  type="button"
                  onClick={() => setText(t => t + (t ? ", " : "") + tag)}
                  className="px-3 py-1.5 bg-accent-lime/8 border border-accent-lime/20 rounded-full text-[12px] text-accent-lime hover:bg-accent-lime/20 transition-all"
                >
                  {tag}
                </button>
              ))}
            </div>

            <button 
              type="submit"
              disabled={loading || !text.trim()}
              className="w-full h-12 bg-accent-lime text-background font-syne font-bold rounded-full hover:brightness-110 disabled:opacity-50 transition-all flex items-center justify-center gap-2"
            >
              {loading ? "🧬 Processing..." : "🧬 Analyze with GNN"}
            </button>
          </form>
        </div>

        {/* Right Card: Graph Visualization */}
        <div className="bg-card border border-accent-lime/20 rounded-[20px] p-8 flex flex-col">
          <span className="block text-[12px] font-medium text-muted uppercase mb-4">Knowledge Graph Traversal</span>
          
          <div className="flex-1 bg-background/50 border border-accent-lime/10 rounded-xl min-h-[240px] relative overflow-hidden flex items-center justify-center">
            <svg width="100%" height="100%" viewBox="0 0 400 300" className="max-w-[320px]">
              {/* Animated Lines */}
              <g className="stroke-muted/30" strokeDasharray="4 4">
                <line x1="200" y1="150" x2="100" y2="80" />
                <line x1="200" y1="150" x2="300" y2="80" />
                <line x1="200" y1="150" x2="100" y2="220" />
                <line x1="200" y1="150" x2="300" y2="220" />
                <line x1="200" y1="150" x2="200" y2="60" />
              </g>

              {/* Nodes */}
              <circle cx="200" cy="150" r="32" className="fill-accent-lime/20 stroke-accent-lime" />
              <text x="200" y="155" textAnchor="middle" className="fill-white font-syne text-[10px] font-bold">Prediction</text>
              
              <circle cx="100" cy="80" r="24" className="fill-accent-teal/10 stroke-accent-teal" />
              <text x="100" y="85" textAnchor="middle" className="fill-muted font-dm-sans text-[8px]">Symptom A</text>
              
              <circle cx="300" cy="80" r="24" className="fill-accent-teal/10 stroke-accent-teal" />
              <text x="300" y="85" textAnchor="middle" className="fill-muted font-dm-sans text-[8px]">Symptom B</text>
              
              <circle cx="100" cy="220" r="20" className="fill-danger/10 stroke-danger" />
              <text x="100" y="225" textAnchor="middle" className="fill-muted font-dm-sans text-[8px]">Avoid</text>
              
              <circle cx="300" cy="220" r="20" className="fill-warning/10 stroke-warning" />
              <text x="300" y="225" textAnchor="middle" className="fill-muted font-dm-sans text-[8px]">Warning</text>

              <circle cx="200" cy="60" r="18" className="fill-accent-lime/5 stroke-accent-lime/30" />
              <text x="200" y="65" textAnchor="middle" className="fill-muted font-dm-sans text-[8px]">BioBERT</text>
            </svg>
          </div>
        </div>
      </div>

      {/* Result Panel */}
      {result && (
        <div className="grid md:grid-cols-2 gap-6 bg-[#0f2920] border border-accent-lime/20 rounded-[20px] p-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          {/* Diagnosis Column */}
          <div className="flex flex-col gap-6">
            <div className="p-6 bg-background/40 rounded-xl border border-accent-lime/10">
              <span className="block text-[11px] font-medium text-muted uppercase mb-2">Predicted Diagnosis</span>
              <h3 className="font-syne font-extrabold text-[22px] text-accent-lime mb-1">{result.disease}</h3>
              <span className="text-[12px] text-muted">ICD-10 Code: {result.icd_code || "R50.9"}</span>
              
              <div className="mt-6 space-y-4">
                {[
                  { label: "GNN Confidence", value: result.confidence * 100 },
                  { label: "Symptom Match", value: 88 },
                ].map((bar, i) => (
                  <div key={i}>
                    <div className="flex justify-between text-[11px] font-medium text-muted mb-1.5">
                      <span>{bar.label}</span>
                      <span>{bar.value.toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-accent-lime to-accent-teal rounded-full transition-all duration-1000"
                        style={{ width: `${bar.value}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="p-6 bg-background/40 rounded-xl border border-accent-lime/10">
              <span className="block text-[10px] font-bold text-accent-lime uppercase mb-3 tracking-wider">GNN Explanation</span>
              <p className="font-dm-sans text-[12px] text-muted leading-relaxed italic">
                {result.explanation}
              </p>
            </div>
          </div>

          {/* Precautions Column */}
          <div className="bg-danger/5 border border-danger/20 rounded-xl p-8 flex flex-col">
            <span className="block text-[11px] font-bold text-danger uppercase mb-6 tracking-wider flex items-center gap-2">
              <span className="text-lg">⚠</span> Foods to Avoid via Neo4j
            </span>
            
            <div className="flex flex-wrap gap-2 mb-8">
              {result.dietary_precautions.map((food: string, i: number) => (
                <div key={i} className="px-3 py-1.5 bg-danger/10 border border-danger/30 rounded-full text-[11px] text-danger/80 flex items-center gap-1.5">
                  <span className="opacity-60">×</span> {food}
                </div>
              ))}
            </div>

            <div className="mt-auto">
              <p className="text-[12px] text-muted mb-4">
                Analysis based on <span className="text-accent-teal">[:CONTRAINDICATED]</span> edges in the Knowledge Graph for <span className="text-accent-lime">{result.disease}</span>.
              </p>
              <button className="w-full py-3 border border-accent-lime/20 text-accent-lime font-syne font-bold text-[12px] rounded-lg hover:bg-accent-lime/5 transition-all">
                View Cypher Query ↗
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

'use client';

export default function Pipeline() {
  const steps = [
    { 
      name: "NER Extraction", 
      desc: "BioBERT extracts medical entities from raw text.", 
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
      ), 
      color: "accent-lime" 
    },
    { 
      name: "Entity Mapping", 
      desc: "Symptoms mapped to Knowledge Graph nodes.", 
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" /><circle cx="12" cy="10" r="3" />
        </svg>
      ), 
      color: "accent-teal" 
    },
    { 
      name: "GNN Inference", 
      desc: "GCN predicts disease from symptom clusters.", 
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2" />
        </svg>
      ), 
      color: "warning" 
    },
    { 
      name: "Graph Reasoning", 
      desc: "Neo4j identifies dietary contraindications.", 
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <path d="M5 3a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2H5zm0 2h14v14H5V5zm4 4v2h2V9H9zm4 0v2h2V9h-2zm-4 4v2h2v-2H9zm4 0v2h2v-2h-2z" />
        </svg>
      ), 
      color: "purple-500" 
    },
    { 
      name: "Final Output", 
      desc: "Explainable result with risk justifications.", 
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="16" y1="13" x2="8" y2="13" /><line x1="16" y1="17" x2="8" y2="17" /><polyline points="10 9 9 9 8 9" />
        </svg>
      ), 
      color: "blue-500" 
    },
  ];

  return (
    <section id="architecture" className="w-full px-8 py-24 bg-card rounded-[24px] mx-auto max-w-[calc(100%-64px)] my-24">
      <div className="max-w-[1100px] mx-auto">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-6 h-[1px] bg-accent-lime"></div>
          <span className="text-[11px] font-medium text-accent-lime uppercase tracking-[0.15em]">System Architecture</span>
        </div>
        <h2 className="font-syne font-extrabold text-4xl text-white mb-16">5-Stage Inference Pipeline</h2>

        <div className="flex flex-col md:flex-row items-center md:items-start justify-between gap-12 relative">
          {steps.map((step, i) => (
            <div key={i} className="flex-1 flex flex-col items-center text-center group">
              <div className={`w-[52px] h-[52px] rounded-full border-[1.5px] border-${step.color} bg-${step.color}/10 flex items-center justify-center text-${step.color} mb-6 relative z-10 group-hover:scale-110 transition-transform`}>
                {step.icon}
              </div>
              <h3 className="font-syne font-bold text-[13px] text-white mb-2">{step.name}</h3>
              <p className="font-dm-sans text-[12px] text-muted leading-relaxed max-w-[160px]">
                {step.desc}
              </p>
              
              {i < steps.length - 1 && (
                <div className="hidden md:block absolute top-[26px] left-[calc(20%*${i}+40px)] w-[calc(20%-80px)] h-[1px] bg-muted/20">
                  <span className="absolute right-0 top-1/2 -translate-y-1/2 text-muted/30 text-[10px]">→</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

'use client';

export default function Pipeline() {
  const steps = [
    { name: "NER Extraction", desc: "BioBERT extracts medical entities from raw text.", icon: "🔍", color: "accent-lime" },
    { name: "Entity Mapping", desc: "Symptoms mapped to Knowledge Graph nodes.", icon: "📍", color: "accent-teal" },
    { name: "GNN Inference", desc: "GCN predicts disease from symptom clusters.", icon: "🧠", color: "warning" },
    { name: "Graph Reasoning", desc: "Neo4j identifies dietary contraindications.", icon: "🕸️", color: "purple-500" },
    { name: "Final Output", desc: "Explainable result with risk justifications.", icon: "📄", color: "blue-500" },
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
              <div className={`w-[52px] h-[52px] rounded-full border-[1.5px] border-${step.color} bg-${step.color}/10 flex items-center justify-center text-xl mb-6 relative z-10 group-hover:scale-110 transition-transform`}>
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

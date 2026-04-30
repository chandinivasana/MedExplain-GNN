'use client';

const features = [
  {
    title: "Graph Convolutional Network",
    desc: "Leveraging PyTorch Geometric to perform inference on local graph neighborhoods for high-accuracy disease prediction.",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="3" />
        <path d="M12 3v3m0 12v3M3 12h3m12 0h3M5.6 5.6l2.1 2.1m8.6 8.6l2.1 2.1M5.6 18.4l2.1-2.1m8.6-8.6l2.1-2.1" />
      </svg>
    ),
    color: "accent-lime",
    tag: "PyG · GCN"
  },
  {
    title: "BioBERT NER Extraction",
    desc: "Surgical precision in extracting symptoms and medical entities from unstructured text using domain-specific Transformers.",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M4.5 7L2 4.5M22 22l-2.5-2.5M2 22l2.5-2.5M22 2l-2.5 2.5M12 2v2M12 20v2M2 12h2M20 12h2M7 7l1.5 1.5M15.5 15.5L17 17M7 17l1.5-1.5M15.5 8.5L17 7" />
        <circle cx="12" cy="12" r="4" />
      </svg>
    ),
    color: "accent-teal",
    tag: "BioBERT · NLP"
  },
  {
    title: "Neo4j Knowledge Graph",
    desc: "A massive, interconnected graph of medical biological knowledge providing the ground truth for dietary contraindications.",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
        <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
        <line x1="12" y1="22.08" x2="12" y2="12" />
      </svg>
    ),
    color: "warning",
    tag: "Neo4j · Cypher"
  },
  {
    title: "Async Redis Queue",
    desc: "Handling high-volume AI inference tasks asynchronously via Redis message passing for a responsive user experience.",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
      </svg>
    ),
    color: "danger",
    tag: "Redis · FastAPI"
  },
  {
    title: "Kubernetes Orchestration",
    desc: "Enterprise-grade scaling with K8s manifests, health probes, and persistent logging across the microservice cluster.",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z" />
        <path d="M12 12l8-4.5M12 12v9M12 12L4 7.5" />
      </svg>
    ),
    color: "blue-500",
    tag: "K8s · Docker"
  },
  {
    title: "Explainability First",
    desc: "Moving beyond black-box AI by providing clear, graph-justified reasons for every medical and nutritional insight.",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
        <circle cx="12" cy="12" r="3" />
      </svg>
    ),
    color: "purple-500",
    tag: "XAI · Graphs"
  }
];

export default function Features() {
  return (
    <section className="max-w-[1100px] mx-auto px-6 py-24">
      {/* Section Label */}
      <div className="flex items-center gap-3 mb-4">
        <div className="w-6 h-[1px] bg-accent-lime"></div>
        <span className="text-[11px] font-medium text-accent-lime uppercase tracking-[0.15em]">Core Capabilities</span>
      </div>
      <h2 className="font-syne font-extrabold text-5xl text-white leading-tight mb-16">
        Built for Production.<br />Designed for Trust.
      </h2>

      <div className="grid md:grid-cols-3 gap-[19px]">
        {features.map((f, i) => (
          <div 
            key={i} 
            className="group relative bg-card border border-accent-lime/12 rounded-[18px] p-[29px] hover:border-accent-lime/30 hover:-translate-y-1 transition-all duration-300 overflow-hidden"
          >
            {/* Bottom Gradient Line on Hover */}
            <div className="absolute bottom-0 left-0 w-full h-[2px] bg-gradient-to-r from-accent-lime/0 via-accent-lime/40 to-accent-teal/0 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            
            {/* Icon Container */}
            <div className={`w-[44px] h-[44px] rounded-xl bg-${f.color}/10 flex items-center justify-center text-${f.color} mb-6`}>
              {f.icon}
            </div>

            <h3 className="font-syne font-bold text-[16px] text-white mb-2">{f.title}</h3>
            <p className="font-dm-sans text-[13px] text-muted leading-[1.7] mb-6">
              {f.desc}
            </p>

            <div className="inline-block px-3 py-1 bg-accent-lime/8 rounded-full">
              <span className="text-[10px] font-bold text-accent-lime uppercase tracking-wider">{f.tag}</span>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

'use client';

const features = [
  {
    title: "Graph Convolutional Network",
    desc: "Leveraging PyTorch Geometric to perform inference on local graph neighborhoods for high-accuracy disease prediction.",
    icon: "🕸️",
    color: "accent-lime",
    tag: "PyG · GCN"
  },
  {
    title: "BioBERT NER Extraction",
    desc: "Surgical precision in extracting symptoms and medical entities from unstructured text using domain-specific Transformers.",
    icon: "🧬",
    color: "accent-teal",
    tag: "BioBERT · NLP"
  },
  {
    title: "Neo4j Knowledge Graph",
    desc: "A massive, interconnected graph of medical biological knowledge providing the ground truth for dietary contraindications.",
    icon: "📂",
    color: "warning",
    tag: "Neo4j · Cypher"
  },
  {
    title: "Async Redis Queue",
    desc: "Handling high-volume AI inference tasks asynchronously via Redis message passing for a responsive user experience.",
    icon: "⚡",
    color: "danger",
    tag: "Redis · FastAPI"
  },
  {
    title: "Kubernetes Orchestration",
    desc: "Enterprise-grade scaling with K8s manifests, health probes, and persistent logging across the microservice cluster.",
    icon: "☸️",
    color: "blue-500",
    tag: "K8s · Docker"
  },
  {
    title: "Explainability First",
    desc: "Moving beyond black-box AI by providing clear, graph-justified reasons for every medical and nutritional insight.",
    icon: "👁️",
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
            <div className={`w-[44px] h-[44px] rounded-xl bg-${f.color}/10 flex items-center justify-center text-[20px] mb-6`}>
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

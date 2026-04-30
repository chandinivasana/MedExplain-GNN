'use client';

const techs = [
  { name: "Next.js (TypeScript)", color: "accent-lime" },
  { name: "FastAPI (Python)", color: "accent-teal" },
  { name: "PyTorch Geometric", color: "pink-500" },
  { name: "BioBERT", color: "blue-400" },
  { name: "Neo4j Graph DB", color: "warning" },
  { name: "Redis Queue", color: "danger" },
  { name: "MongoDB Atlas", color: "green-500" },
  { name: "Kubernetes", color: "indigo-500" },
  { name: "Docker", color: "blue-600" },
  { name: "Tailwind CSS", color: "sky-400" },
  { name: "HuggingFace", color: "yellow-400" },
  { name: "Makefile CI/CD", color: "muted" },
];

export default function TechStack() {
  return (
    <section className="max-w-[1100px] mx-auto px-6 py-24">
      {/* Section Label */}
      <div className="flex items-center gap-3 mb-4">
        <div className="w-6 h-[1px] bg-accent-lime"></div>
        <span className="text-[11px] font-medium text-accent-lime uppercase tracking-[0.15em]">Tech Stack</span>
      </div>
      <h2 className="font-syne font-extrabold text-5xl text-white mb-12">Built on the Best.</h2>

      <div className="flex flex-wrap gap-[10px] mb-16">
        {techs.map((t, i) => (
          <div 
            key={i}
            className="group flex items-center gap-2.5 bg-card border border-accent-lime/12 rounded-full py-2 px-4.5 hover:border-accent-lime/30 transition-all cursor-default"
          >
            <div className={`w-1.5 h-1.5 rounded-full bg-${t.color}`}></div>
            <span className="font-dm-sans text-[13px] text-white group-hover:text-accent-lime transition-colors">
              {t.name}
            </span>
          </div>
        ))}
      </div>

      <div className="flex justify-center">
        <button className="h-12 px-10 bg-accent-lime text-background font-syne font-bold rounded-full hover:brightness-110 transition-all shadow-[0_0_20px_rgba(168,255,62,0.15)]">
          🚀 Generate Infrastructure Code
        </button>
      </div>
    </section>
  );
}

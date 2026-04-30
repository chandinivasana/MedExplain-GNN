'use client';

export default function Hero() {
  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center pt-32 pb-16 overflow-hidden">
      {/* Background Layers */}
      <div className="absolute inset-0 z-0">
        <div className="absolute inset-0 bg-grid opacity-30"></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-accent-lime/6 rounded-full blur-[120px]"></div>
        <div className="absolute top-0 right-0 w-[400px] h-[400px] bg-accent-teal/5 rounded-full blur-[100px]"></div>
        <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-green-900/4 rounded-full blur-[110px]"></div>
      </div>

      {/* Content */}
      <div className="relative z-10 flex flex-col items-center text-center px-6">
        {/* Badge */}
        <div className="flex items-center gap-3 px-4 py-2 bg-accent-lime/10 border border-accent-lime/20 rounded-full mb-8">
          <div className="w-1.5 h-1.5 bg-accent-lime rounded-full animate-pulse-lime"></div>
          <span className="text-[12px] font-medium text-accent-lime uppercase tracking-[0.08em]">
            Graph Neural Network · Medical AI
          </span>
        </div>

        {/* Heading */}
        <h1 className="font-syne font-extrabold text-6xl md:text-8xl lg:text-[72px] leading-[1.0] tracking-[-0.04em] mb-6">
          <span className="block text-white">Diagnose.</span>
          <span className="block text-accent-lime">Explain.</span>
          <span className="block text-accent-teal">Trust.</span>
        </h1>

        {/* Subtitle */}
        <p className="max-w-[560px] font-dm-sans text-[17px] text-muted leading-[1.7] mb-12">
          A transparent, graph-powered medical reasoning engine. From raw symptom text to explainable disease prediction — with dietary contraindications backed by biological knowledge graphs.
        </p>

        {/* Buttons */}
        <div className="flex items-center gap-4 mb-16">
          <button className="h-12 px-8 bg-accent-lime text-background font-syne font-bold rounded-full hover:brightness-110 transition-all">
            ⚡ Try the Engine
          </button>
          <button className="h-12 px-8 border border-white/15 text-white font-syne font-bold rounded-full hover:bg-white/5 transition-all">
            ◎ View Architecture
          </button>
        </div>

        {/* Stats Bar */}
        <div className="w-full max-w-[900px] grid grid-cols-2 md:grid-cols-4 bg-card border border-accent-lime/12 rounded-2xl overflow-hidden divide-x divide-accent-lime/12">
          {[
            { value: "94.2%", label: "Prediction Accuracy" },
            { value: "50k+", label: "Medical Entities" },
            { value: "12ms", label: "Inference Latency" },
            { value: "100%", label: "Explainable Path" },
          ].map((stat, i) => (
            <div key={i} className="py-6 flex flex-col items-center text-center">
              <span className="font-syne font-extrabold text-[28px] text-accent-lime">{stat.value}</span>
              <span className="font-dm-sans text-[11px] uppercase text-muted tracking-[0.08em]">{stat.label}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

'use client';

export default function Footer() {
  return (
    <footer className="w-full border-t border-accent-lime/12 pt-12 pb-16 mt-24">
      <div className="max-w-[1100px] mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-8">
        {/* Left: Logo Group */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-lime to-accent-teal flex items-center justify-center">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="black" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2v20M2 12h20M7 7l10 10M17 7L7 10" />
            </svg>
          </div>
          <span className="font-syne font-extrabold text-[17px] text-white tracking-tight">MedExplain-GNN</span>
        </div>

        {/* Center: Links */}
        <div className="flex items-center gap-8">
          {["Demo", "Architecture"].map(link => (
            <a key={link} href={`#${link.toLowerCase()}`} className="font-dm-sans text-[13px] text-muted hover:text-accent-lime transition-colors">
              {link}
            </a>
          ))}
        </div>

        {/* Right: Copyright */}
        <div className="font-dm-sans text-[13px] text-muted">
          © 2025 MedExplain-GNN · MIT License
        </div>
      </div>
    </footer>
  );
}

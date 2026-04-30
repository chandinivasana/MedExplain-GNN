'use client';

export default function Navbar() {
  return (
    <nav className="fixed top-0 left-0 w-full h-16 z-50 bg-background/80 backdrop-blur-xl border-b border-accent-lime/12">
      <div className="max-w-7xl mx-auto px-6 h-full flex items-center justify-between">
        {/* Left: Logo */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-lime to-accent-teal flex items-center justify-center">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="black" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2v20M2 12h20M7 7l10 10M17 7L7 10" />
            </svg>
          </div>
          <span className="font-syne font-extrabold text-[17px] text-white tracking-tight">MedExplain-GNN</span>
        </div>

        {/* Center: Links */}
        <div className="hidden md:flex items-center gap-8">
          {["Demo", "Architecture"].map((link) => (
            <a key={link} href={`#${link.toLowerCase()}`} className="font-dm-sans text-sm text-muted hover:text-accent-lime transition-colors">
              {link}
            </a>
          ))}
        </div>

        {/* Right: Empty for balance or could add social/github */}
        <div className="w-[120px] hidden md:block"></div>
      </div>
    </nav>
  );
}

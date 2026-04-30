'use client';

import Navbar from '@/components/Navbar';
import Hero from '@/components/Hero';
import LiveDemo from '@/components/LiveDemo';
import Pipeline from '@/components/Pipeline';
import Features from '@/components/Features';
import TechStack from '@/components/TechStack';
import Footer from '@/components/Footer';

export default function Home() {
  return (
    <main className="min-h-screen bg-background text-foreground selection:bg-accent-lime/30 selection:text-white">
      <Navbar />
      
      {/* Hero Section */}
      <Hero />

      {/* Live Demo Section */}
      <LiveDemo />

      {/* Pipeline Visualization */}
      <Pipeline />

      {/* Features Grid */}
      <Features />

      {/* Tech Stack Chips */}
      <TechStack />

      {/* Site Footer */}
      <Footer />
    </main>
  );
}

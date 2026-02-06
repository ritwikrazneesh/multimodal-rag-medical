"""
MedlinePlus Medical Encyclopedia Crawler
Downloads all encyclopedia articles with deduplication
"""

import os
import json
import time
import hashlib
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime


class MedlinePlusEncyclopediaCrawler:
    """Crawl MedlinePlus Medical Encyclopedia"""
    
    def __init__(self, output_dir="data/medlineplus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.output_dir / "metadata.json"
        self.metadata = self.load_metadata()
        
        # Track content hashes for deduplication
        self.content_hashes = set()
        self.load_existing_hashes()
        
        self.base_url = "https://medlineplus.gov/ency/article"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def load_metadata(self):
        """Load existing metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_metadata(self):
        """Save metadata"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def load_existing_hashes(self):
        """Load hashes of already-downloaded content"""
        for meta in self.metadata.values():
            if 'content_hash' in meta:
                self.content_hashes.add(meta['content_hash'])
    
    def compute_content_hash(self, content):
        """Compute SHA256 hash of content for deduplication"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def fetch_article(self, article_id):
        """
        Fetch a single encyclopedia article
        
        Args:
            article_id: 6-digit string like "000313"
        
        Returns:
            dict with article data or None if failed/duplicate
        """
        url = f"{self.base_url}/{article_id}.htm"
        
        try:
            response = self.session.get(url, timeout=10)
            
            # Handle 404s silently
            if response.status_code == 404:
                return None
            
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1', class_='with-also')
            if not title_elem:
                title_elem = soup.find('h1')
            
            if not title_elem:
                return None
            
            title = title_elem.get_text(strip=True)
            
            # Extract main article content
            article_body = soup.find('div', id='ency_summary')
            if not article_body:
                article_body = soup.find('article')
            
            if not article_body:
                return None
            
            # Remove unwanted elements
            for unwanted in article_body.find_all([
                'script', 'style', 'nav', 'aside', 'footer', 
                'header', 'iframe', 'noscript', 'button'
            ]):
                unwanted.decompose()
            
            # Remove UI elements
            for elem in article_body.find_all(class_=[
                'page-actions', 'section-nav', 'social-share',
                'breadcrumbs', 'share-bar'
            ]):
                elem.decompose()
            
            # Extract clean text
            content = article_body.get_text(separator='\n', strip=True)
            content = self.clean_content(content)
            
            # Validate content
            if len(content) < 100:  # Too short
                return None
            
            # Check for duplicate content
            content_hash = self.compute_content_hash(content)
            if content_hash in self.content_hashes:
                return None  # Duplicate
            
            # Extract last updated date
            date_elem = soup.find('meta', {'name': 'date'})
            if date_elem and date_elem.has_attr('content'):
                date_updated = date_elem['content']
            else:
                # Try to find "Reviewed" text
                review_text = soup.find(string=lambda s: s and 'Reviewed' in s)
                date_updated = review_text.strip() if review_text else datetime.now().strftime('%Y-%m-%d')
            
            # Mark as seen
            self.content_hashes.add(content_hash)
            
            return {
                'id': article_id,
                'title': title,
                'content': content,
                'url': url,
                'source': 'MedlinePlus Encyclopedia',
                'organization': 'NIH - National Library of Medicine',
                'date_updated': date_updated,
                'date_fetched': datetime.now().isoformat(),
                'word_count': len(content.split()),
                'content_hash': content_hash
            }
            
        except requests.RequestException:
            return None
        except Exception as e:
            print(f"  ⚠ Error parsing {article_id}: {e}")
            return None
    
    def clean_content(self, text):
        """Clean extracted content"""
        # Split into lines
        lines = text.split('\n')
        
        # Remove empty and very short lines
        lines = [line.strip() for line in lines if len(line.strip()) > 2]
        
        # Remove exact duplicates while preserving order
        seen = set()
        unique_lines = []
        for line in lines:
            line_lower = line.lower()
            if line_lower not in seen:
                seen.add(line_lower)
                unique_lines.append(line)
        
        # Join with double newlines
        text = '\n\n'.join(unique_lines)
        
        # Remove common UI text
        ui_phrases = [
            'Skip to main content',
            'You Are Here:',
            'Print this page',
            'Share this page',
            'Email this page',
            'Add to My Med List',
            'Images',
            'References',
        ]
        
        for phrase in ui_phrases:
            text = text.replace(phrase, '')
        
        # Remove excessive whitespace
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        
        return text.strip()
    
    def save_article(self, article_data):
        """Save article to file"""
        # Create safe filename from ID and title
        safe_title = "".join(
            c for c in article_data['title'][:50]
            if c.isalnum() or c in (' ', '-', '_')
        ).strip()
        
        filename = f"{article_data['id']}_{safe_title}.txt"
        filepath = self.output_dir / filename
        
        # Save content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(article_data['content'])
        
        # Save metadata
        self.metadata[str(filepath)] = {
            'id': article_data['id'],
            'title': article_data['title'],
            'url': article_data['url'],
            'source': article_data['source'],
            'organization': article_data['organization'],
            'date_updated': article_data['date_updated'],
            'date_fetched': article_data['date_fetched'],
            'word_count': article_data['word_count'],
            'content_hash': article_data['content_hash'],
            'filepath': str(filepath)
        }
        
        return filepath
    
    def crawl_encyclopedia(self, start_id=1, end_id=7999, delay=1.0, save_interval=50):
        """
        Crawl the entire encyclopedia
        
        Args:
            start_id: Starting article ID (default: 1)
            end_id: Ending article ID (default: 7999)
            delay: Delay between requests in seconds
            save_interval: Save metadata every N articles
        """
        print(f"\n{'='*70}")
        print(f"📚 MedlinePlus Encyclopedia Crawler")
        print(f"{'='*70}")
        print(f"Range: {start_id:06d} to {end_id:06d} ({end_id - start_id + 1:,} IDs)")
        print(f"Delay: {delay}s between requests")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\n")
        
        successful = 0
        failed = 0
        duplicates = 0
        start_time = time.time()
        
        for article_num in range(start_id, end_id + 1):
            article_id = f"{article_num:06d}"
            
            # Progress indicator
            if article_num % 100 == 0:
                elapsed = time.time() - start_time
                rate = (article_num - start_id + 1) / elapsed if elapsed > 0 else 0
                remaining = (end_id - article_num) / rate if rate > 0 else 0
                print(f"\n[Progress] ID {article_id} | ✓ {successful} | ✗ {failed} | ⊗ {duplicates}")
                print(f"           Rate: {rate:.1f} articles/min | ETA: {remaining/60:.1f} min\n")
            
            # Fetch article
            article = self.fetch_article(article_id)
            
            if article:
                # Check if duplicate by hash (already done in fetch_article)
                filepath = self.save_article(article)
                successful += 1
                print(f"✓ {article_id} | {article['title'][:50]} ({article['word_count']} words)")
            else:
                failed += 1
                if failed % 50 == 0:  # Only print every 50 failures to reduce spam
                    print(f"✗ {article_id} (not found or duplicate)")
            
            # Save metadata periodically
            if successful > 0 and successful % save_interval == 0:
                self.save_metadata()
                print(f"💾 Saved metadata ({successful} articles so far)")
            
            # Rate limiting
            time.sleep(delay)
        
        # Final save
        self.save_metadata()
        
        # Summary
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✅ CRAWL COMPLETE")
        print(f"{'='*70}")
        print(f"�� Successfully fetched: {successful:,} articles")
        print(f"✗ Failed (404/errors):  {failed:,}")
        print(f"📁 Saved to: {self.output_dir}")
        print(f"📋 Metadata: {self.metadata_file}")
        print(f"⏱  Time elapsed: {elapsed/60:.1f} minutes")
        print(f"📊 Average: {successful/elapsed*60:.1f} articles/minute")
        print(f"💾 Total size: ~{successful * 2:.0f} KB")
        print(f"{'='*70}\n")
        
        return successful, failed


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Crawl MedlinePlus Medical Encyclopedia'
    )
    parser.add_argument(
        '--start', type=int, default=1,
        help='Starting article ID (default: 1)'
    )
    parser.add_argument(
        '--end', type=int, default=7999,
        help='Ending article ID (default: 7999)'
    )
    parser.add_argument(
        '--delay', type=float, default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--output', type=str, default='data/medlineplus',
        help='Output directory (default: data/medlineplus)'
    )
    
    args = parser.parse_args()
    
    crawler = MedlinePlusEncyclopediaCrawler(output_dir=args.output)
    crawler.crawl_encyclopedia(
        start_id=args.start,
        end_id=args.end,
        delay=args.delay
    )


if __name__ == "__main__":
    main()
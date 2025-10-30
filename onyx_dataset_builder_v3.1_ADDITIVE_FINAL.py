"""
Onyx Coffee Lab Dataset Builder - V3.0 ENHANCED
Phase 1 & Phase 2 Feature Extraction for RoastFormer Conditioning
Extracts: Origin, Process, Variety, Altitude, Roast Level, Machine, Drying Method, etc.
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import json
import time
import re
import pandas as pd
from datetime import datetime
import os
from pathlib import Path

class OnyxDatasetBuilderV3:
    """
    Enhanced dataset builder with comprehensive feature extraction
    For transformer conditioning on bean characteristics
    """
    
    def __init__(self, output_dir=None, use_date_suffix=True):
        """
        Initialize dataset builder with automatic date-stamped directories
        
        Args:
            output_dir: Base directory name (default: 'onyx_dataset')
            use_date_suffix: If True, automatically append date (e.g., 'onyx_dataset_2024_10_28')
        """
        # Set base directory
        base_dir = output_dir if output_dir else 'onyx_dataset'
        
        # Add date suffix if enabled
        if use_date_suffix:
            date_suffix = datetime.now().strftime('%Y_%m_%d')
            self.output_dir = f"{base_dir}_{date_suffix}"
        else:
            self.output_dir = base_dir
        
        self.base_url = "https://onyxcoffeelab.com"
        
        # Create output directory structure
        os.makedirs(f"{self.output_dir}/profiles", exist_ok=True)
        os.makedirs(f"{self.output_dir}/profiles_by_batch", exist_ok=True)
        os.makedirs(f"{self.output_dir}/metadata", exist_ok=True)
        os.makedirs(f"{self.output_dir}/logs", exist_ok=True)
        
        # Load batch history from ALL previous scrapes
        self.batch_history = self._load_global_batch_history()
        
        print(f"✓ Dataset directory: {self.output_dir}/")
        print("✓ ADDITIVE mode: Date-stamped, won't overwrite previous scrapes")
        print(f"✓ Batch history loaded: {len(self.batch_history)} products tracked")
        print("✓ Enhanced feature extraction: Phase 1 + Phase 2 + Flavors")
    
    def _load_global_batch_history(self):
        """
        Load batch history from all previous scrape directories
        This allows checking if we already have a batch across all dates
        """
        batch_history = {}
        
        # Look for all onyx_dataset_* directories
        parent_dir = os.path.dirname(os.path.abspath(self.output_dir)) or '.'
        pattern = 'onyx_dataset_*'
        
        # Find all existing dataset directories
        import glob
        existing_dirs = glob.glob(os.path.join(parent_dir, pattern))
        
        if not existing_dirs:
            print("  No previous scrapes found - starting fresh")
            return batch_history
        
        print(f"  Found {len(existing_dirs)} previous scrape(s)")
        
        # Load batch info from each directory
        for dir_path in existing_dirs:
            dataset_file = os.path.join(dir_path, 'complete_dataset.json')
            
            if not os.path.exists(dataset_file):
                continue
            
            try:
                with open(dataset_file, 'r') as f:
                    dataset = json.load(f)
                
                # Extract batch info from profiles
                for profile in dataset.get('profiles', []):
                    product_name = profile['metadata'].get('product_name')
                    batch_num = profile['metadata'].get('roast_info', {}).get('batch')
                    roast_date = profile['metadata'].get('roast_info', {}).get('roast_date')
                    
                    if not product_name:
                        continue
                    
                    # Initialize product history
                    if product_name not in batch_history:
                        batch_history[product_name] = []
                    
                    # Add batch record
                    batch_history[product_name].append({
                        'batch_number': batch_num,
                        'roast_date': roast_date,
                        'scraped_at': profile.get('scraped_at'),
                        'source_dir': os.path.basename(dir_path)
                    })
            
            except Exception as e:
                print(f"  Warning: Could not load {dataset_file}: {e}")
                continue
        
        return batch_history
    
    def _check_if_new_batch(self, product_name, current_batch_num, current_roast_date):
        """
        Check if this is a new batch we haven't scraped before
        
        Returns:
            (is_new: bool, reason: str)
        """
        # If we've never seen this product, it's new
        if product_name not in self.batch_history:
            return True, "New product - never scraped"
        
        # Get all batches we've seen for this product
        previous_batches = self.batch_history[product_name]
        
        # Check if we've seen this exact batch number
        if current_batch_num:
            for prev in previous_batches:
                if prev['batch_number'] == current_batch_num:
                    return False, f"Already have batch #{current_batch_num}"
        
        # Check if we've seen this exact roast date
        if current_roast_date:
            for prev in previous_batches:
                if prev['roast_date'] == current_roast_date:
                    return False, f"Already have roast from {current_roast_date}"
        
        # If we can't determine batch/date, consider it new to be safe
        if not current_batch_num and not current_roast_date:
            return True, "No batch/date info - scraping to be safe"
        
        # New batch!
        return True, f"New batch (previous: {len(previous_batches)} batches)"
    
    def setup_driver(self):
        """Setup Chrome driver with options"""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        driver = webdriver.Chrome(options=options)
        return driver
    
    def find_all_product_urls(self):
        """
        Step 1: Find all coffee product URLs from Onyx catalog
        """
        print("\n" + "="*80)
        print("STEP 1: Discovering all Onyx coffee products")
        print("="*80)
        
        collection_url = f"{self.base_url}/collections/coffee"
        all_product_urls = set()
        
        driver = self.setup_driver()
        
        try:
            print(f"\nScanning: {collection_url}")
            driver.get(collection_url)
            time.sleep(3)
            
            # Scroll to load all products
            print("  Scrolling to load all products...")
            last_height = driver.execute_script("return document.body.scrollHeight")
            scroll_attempts = 0
            max_scrolls = 10
            
            while scroll_attempts < max_scrolls:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                new_height = driver.execute_script("return document.body.scrollHeight")
                
                if new_height == last_height:
                    break
                    
                last_height = new_height
                scroll_attempts += 1
                print(f"    Scroll {scroll_attempts}/{max_scrolls}...")
            
            print("  ✓ Page fully loaded")
            
            # Parse the page
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            product_links = soup.find_all('a', href=re.compile(r'/products/[^/]+'))
            
            print(f"  Found {len(product_links)} product links")
            
            for link in product_links:
                href = link.get('href')
                if href and '/products/' in href:
                    full_url = f"{self.base_url}{href}" if href.startswith('/') else href
                    clean_url = full_url.split('?')[0].split('#')[0]
                    all_product_urls.add(clean_url)
            
            product_list = sorted(list(all_product_urls))
            
            print(f"\n✓ Total unique products discovered: {len(product_list)}")
            
            # Save product list
            with open(f"{self.output_dir}/product_urls.json", 'w') as f:
                json.dump(product_list, f, indent=2)
            
            print(f"✓ Product URLs saved to: {self.output_dir}/product_urls.json")
            
            with open(f"{self.output_dir}/product_urls.txt", 'w') as f:
                for i, url in enumerate(product_list, 1):
                    product_name = url.split('/')[-1].replace('-', ' ').title()
                    f.write(f"{i:2d}. {product_name:40s} {url}\n")
            
            return product_list
            
        except Exception as e:
            print(f"✗ Error discovering products: {e}")
            import traceback
            traceback.print_exc()
            return []
            
        finally:
            driver.quit()
    
    def extract_enhanced_metadata(self, driver, url):
        """
        ENHANCED: Extract comprehensive metadata for transformer conditioning
        Phase 1: Origin, Process, Target Roast Level, Target Finish Temp
        Phase 2: Variety, Altitude, Bean Density (proxy), Drying Method
        """
        metadata = {
            'url': url,
            'product_name': None,
            
            # PHASE 1: Critical Features (always needed)
            'origin': None,  # Colombia, Ethiopia, Kenya, etc.
            'process': None,  # Washed, Natural, Honey, Anaerobic
            'roast_level': None,  # Expressive Light, Medium, Dark
            'roast_level_agtron': None,  # Agtron #135 numeric value
            'target_finish_temp': None,  # Inferred from roast level
            
            # PHASE 2: Helpful Features (improve quality)
            'variety': None,  # Mixed, Heirloom, Caturra, Bourbon
            'altitude': None,  # 1500 MASL, 1800-2200m
            'altitude_numeric': None,  # Average altitude in meters
            'bean_density_proxy': None,  # Calculated from altitude
            'drying_method': None,  # Raised-Bed, Patio, African Bed
            
            # Additional Context
            'harvest_season': None,  # Rotating Microlots, October, etc.
            'roaster_machine': None,  # Loring S70 Peregrine
            'preferred_extraction': None,  # Filter, Espresso, Both
            'caffeine_mg': None,  # 215mg
            
            # FLAVOR NOTES (NEW!)
            'flavor_notes_raw': None,  # "BERRIES STONE FRUIT EARL GREY HONEYSUCKLE ROUND"
            'flavor_notes_parsed': None,  # ['BERRIES', 'STONE', 'FRUIT', 'EARL', 'GREY', ...]
            'flavor_categories': None,  # ['fruity', 'tea', 'floral', 'body']
            
            # Roast-specific metadata
            'roast_info': {}
        }
        
        try:
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            page_text = soup.get_text()
            
            # Product name
            title = (soup.find('h1', class_=re.compile(r'product.*title', re.I)) or
                    soup.find('h1', class_=re.compile(r'title', re.I)) or
                    soup.find('h1'))
            
            if title:
                metadata['product_name'] = title.get_text().strip()
            else:
                metadata['product_name'] = url.split('/')[-1].replace('-', ' ').title()
            
            # ===== PHASE 1 FEATURES =====
            
            # 1. ORIGIN - "Colombia, Ethiopia"
            origin_patterns = [
                r'ORIGIN[:\s]*([^\n]+)',
                r'origin[:\s]*([^\n]+)',
            ]
            for pattern in origin_patterns:
                match = re.search(pattern, page_text, re.I)
                if match:
                    origin_text = match.group(1).strip()
                    # Clean up common suffixes
                    origin_text = re.sub(r'\s*(VARIETY|PROCESS|HARVEST).*', '', origin_text, flags=re.I)
                    metadata['origin'] = origin_text[:100]
                    break
            
            # 2. PROCESS - "Washed", "Natural", etc.
            process_keywords = ['washed', 'natural', 'honey', 'anaerobic', 'wet hulled', 
                              'pulped natural', 'semi-washed', 'experimental']
            page_text_lower = page_text.lower()
            for keyword in process_keywords:
                if keyword in page_text_lower:
                    metadata['process'] = keyword.title()
                    break
            
            # 3. ROAST LEVEL - "Expressive Light Agtron #135"
            roast_level_patterns = [
                r'(Expressive\s+Light|Expressive\s+Dark|Light|Medium|Dark|Full\s+City)',
                r'ROAST\s+LEVEL[:\s]*([^\n]+)',
            ]
            for pattern in roast_level_patterns:
                match = re.search(pattern, page_text, re.I)
                if match:
                    roast_text = match.group(1).strip()
                    # Remove Agtron number if present
                    roast_clean = re.sub(r'Agtron.*', '', roast_text, flags=re.I).strip()
                    metadata['roast_level'] = roast_clean[:50]
                    break
            
            # 4. AGTRON NUMBER - "#135"
            agtron_match = re.search(r'Agtron\s*#?(\d+)', page_text, re.I)
            if agtron_match:
                metadata['roast_level_agtron'] = int(agtron_match.group(1))
            
            # 5. TARGET FINISH TEMP - Infer from Agtron or roast level
            metadata['target_finish_temp'] = self._infer_finish_temp(
                metadata['roast_level'], 
                metadata['roast_level_agtron']
            )
            
            # ===== PHASE 2 FEATURES =====
            
            # 6. VARIETY - "Mixed", "Heirloom", "Caturra"
            variety_patterns = [
                r'VARIETY[:\s]*([^\n]+)',
                r'Variet(?:y|ies)[:\s]*([^\n]+)',
            ]
            for pattern in variety_patterns:
                match = re.search(pattern, page_text, re.I)
                if match:
                    variety_text = match.group(1).strip()
                    # Clean up
                    variety_text = re.sub(r'\s*(HARVEST|PROCESS|ORIGIN).*', '', variety_text, flags=re.I)
                    metadata['variety'] = variety_text[:100]
                    break
            
            # 7. ALTITUDE - "1500 MASL" or "1800-2200m"
            altitude_patterns = [
                r'ELEVATION[:\s]*(\d+(?:-\d+)?)\s*(?:MASL|m|meters)',
                r'(\d+(?:-\d+)?)\s*(?:MASL|m|meters)',
                r'altitude[:\s]*(\d+(?:-\d+)?)\s*(?:MASL|m|meters)',
            ]
            for pattern in altitude_patterns:
                match = re.search(pattern, page_text, re.I)
                if match:
                    altitude_str = match.group(1)
                    metadata['altitude'] = f"{altitude_str} MASL"
                    
                    # Calculate numeric average
                    if '-' in altitude_str:
                        low, high = map(int, altitude_str.split('-'))
                        metadata['altitude_numeric'] = (low + high) / 2
                    else:
                        metadata['altitude_numeric'] = int(altitude_str)
                    
                    # 8. BEAN DENSITY PROXY - Higher altitude = denser beans
                    # Rough formula: density increases ~0.05 g/cm³ per 1000m
                    # Base density ~0.65 g/cm³ at sea level
                    if metadata['altitude_numeric']:
                        base_density = 0.65
                        altitude_km = metadata['altitude_numeric'] / 1000.0
                        metadata['bean_density_proxy'] = base_density + (altitude_km * 0.05)
                    break
            
            # 9. DRYING METHOD - "Raised-Bed Dried"
            drying_patterns = [
                r'(Raised-Bed|Patio|African\s+Bed|Mechanical|Sun)\s*Dried?',
                r'DRYING\s+METHOD[:\s]*([^\n]+)',
            ]
            for pattern in drying_patterns:
                match = re.search(pattern, page_text, re.I)
                if match:
                    drying_text = match.group(1).strip()
                    metadata['drying_method'] = drying_text[:50]
                    break
            
            # ===== ADDITIONAL CONTEXT =====
            
            # 10. HARVEST SEASON - "Rotating Microlots"
            harvest_patterns = [
                r'HARVEST\s+SEASON[:\s]*([^\n]+)',
                r'Harvest[:\s]*([^\n]+)',
            ]
            for pattern in harvest_patterns:
                match = re.search(pattern, page_text, re.I)
                if match:
                    harvest_text = match.group(1).strip()
                    # Clean up (remove process info if included)
                    harvest_text = re.sub(r'\s*Wa\s*$', '', harvest_text)  # Remove trailing "Wa"
                    metadata['harvest_season'] = harvest_text[:100]
                    break
            
            # 11. ROASTER MACHINE - "Loring S70 Peregrine"
            roaster_patterns = [
                r'(Loring|Probat|Diedrich|Giesen)\s+[A-Z0-9\s]+(?:Peregrine)?',
                r'PRODUCTION\s+ROASTER[:\s]*([^\n]+)',
            ]
            for pattern in roaster_patterns:
                match = re.search(pattern, page_text, re.I)
                if match:
                    roaster_text = match.group(0).strip() if 'Loring' in pattern else match.group(1).strip()
                    metadata['roaster_machine'] = roaster_text[:100]
                    break
            
            # 12. PREFERRED EXTRACTION - "Filter & Espresso"
            extraction_patterns = [
                r'(Filter\s*&\s*Espresso|Filter|Espresso|Omni)',
                r'PREFERRED\s+EXTRACTION[:\s]*([^\n]+)',
            ]
            for pattern in extraction_patterns:
                match = re.search(pattern, page_text, re.I)
                if match:
                    extraction_text = match.group(1).strip()
                    metadata['preferred_extraction'] = extraction_text[:50]
                    break
            
            # 13. CAFFEINE - "215mg"
            caffeine_match = re.search(r'(\d+)\s*mg', page_text, re.I)
            if caffeine_match:
                metadata['caffeine_mg'] = int(caffeine_match.group(1))
            
            # 14. FLAVOR NOTES - "BERRIES STONE FRUIT EARL GREY HONEYSUCKLE ROUND"
            # Pattern: All-caps text right before "Filter & Espresso" or "PREFERRED EXTRACTION"
            flavor_patterns = [
                r'([A-Z][A-Z\s]{15,150}?)\s*(?:Filter|PREFERRED\s+EXTRACTION)',
                r'geometry\s*\n\s*([A-Z][A-Z\s]{15,150}?)\s*Filter',
            ]
            
            for pattern in flavor_patterns:
                flavor_match = re.search(pattern, page_text, re.MULTILINE)
                if flavor_match:
                    flavor_text = flavor_match.group(1).strip()
                    
                    # Clean up - remove product name if present
                    flavor_text = re.sub(r'^\w+\s*$', '', flavor_text, flags=re.MULTILINE).strip()
                    
                    # Store raw flavor text
                    metadata['flavor_notes_raw'] = flavor_text
                    
                    # Parse into individual notes
                    flavor_words = flavor_text.split()
                    # Filter out very short words and common non-flavor terms
                    flavor_notes = [w for w in flavor_words if len(w) > 2 and w not in ['THE', 'AND', 'OR', 'WITH']]
                    metadata['flavor_notes_parsed'] = flavor_notes
                    
                    # Categorize flavors
                    metadata['flavor_categories'] = self._categorize_flavors(flavor_notes)
                    break
            
            # Legacy roast info patterns (batch, date, etc.)
            roast_patterns = {
                'batch': r'batch[:\s#]*(\d+)',
                'roast_date': r'roast\s*date[:\s]*([^\n]{5,30})',
                'charge_temp': r'charge\s*temp[:\s]*(\d+\.?\d*)',
                'duration': r'duration[:\s]*(\d+)',
            }
            
            for key, pattern in roast_patterns.items():
                match = re.search(pattern, page_text, re.I)
                if match:
                    metadata['roast_info'][key] = match.group(1).strip()
            
            # Print extraction summary
            print(f"    📝 Extracted metadata:")
            print(f"       Origin: {metadata['origin']}")
            print(f"       Process: {metadata['process']}")
            print(f"       Roast Level: {metadata['roast_level']} (Agtron: {metadata['roast_level_agtron']})")
            print(f"       Variety: {metadata['variety']}")
            print(f"       Altitude: {metadata['altitude']} ({metadata['altitude_numeric']}m)")
            print(f"       Drying: {metadata['drying_method']}")
            print(f"       Flavors: {metadata['flavor_notes_raw']}")
            print(f"       Flavor Categories: {metadata['flavor_categories']}")
            
        except Exception as e:
            print(f"    ⚠️  Warning: Metadata extraction incomplete - {e}")
        
        return metadata
    
    def _categorize_flavors(self, flavor_notes):
        """
        Categorize flavor notes into flavor families
        
        Args:
            flavor_notes: List of flavor words (e.g., ['BERRIES', 'CHOCOLATE', 'FLORAL'])
        
        Returns:
            List of flavor categories present
        """
        categories = []
        
        # Flavor category mappings
        flavor_categories = {
            'fruity': ['BERRY', 'BERRIES', 'FRUIT', 'CHERRY', 'CITRUS', 'APPLE', 
                      'STONE', 'TROPICAL', 'GRAPE', 'PLUM', 'PEACH', 'APRICOT',
                      'BLUEBERRY', 'STRAWBERRY', 'RASPBERRY', 'BLACKBERRY'],
            'floral': ['FLORAL', 'HONEYSUCKLE', 'JASMINE', 'ROSE', 'LAVENDER', 
                      'HIBISCUS', 'CHAMOMILE'],
            'chocolate': ['CHOCOLATE', 'COCOA', 'CACAO', 'DARK CHOCOLATE', 'MILK CHOCOLATE'],
            'nutty': ['NUT', 'NUTS', 'ALMOND', 'HAZELNUT', 'PECAN', 'WALNUT', 'PEANUT'],
            'caramel': ['CARAMEL', 'TOFFEE', 'BROWN SUGAR', 'MOLASSES', 'BUTTERSCOTCH'],
            'spice': ['SPICE', 'SPICY', 'CINNAMON', 'CLOVE', 'CARDAMOM', 'NUTMEG', 'GINGER'],
            'tea': ['TEA', 'EARL GREY', 'BLACK TEA', 'GREEN TEA', 'BERGAMOT'],
            'body': ['ROUND', 'CREAMY', 'SILKY', 'SMOOTH', 'SYRUPY', 'BUTTERY', 'VELVETY'],
            'citrus': ['CITRUS', 'LEMON', 'ORANGE', 'LIME', 'GRAPEFRUIT', 'TANGERINE'],
            'herbal': ['HERBAL', 'MINT', 'SAGE', 'THYME', 'BASIL'],
            'sweet': ['SWEET', 'HONEY', 'SUGAR', 'VANILLA', 'MARSHMALLOW'],
        }
        
        # Check each flavor note against categories
        for note in flavor_notes:
            note_upper = note.upper()
            for category, keywords in flavor_categories.items():
                if any(keyword in note_upper for keyword in keywords):
                    if category not in categories:
                        categories.append(category)
        
        return categories
    
    def _infer_finish_temp(self, roast_level, agtron_number):
        """
        Infer target finish temperature from roast level or Agtron number
        Agtron scale: Higher number = lighter roast = lower temp
        """
        # If we have Agtron number, use that
        if agtron_number:
            # Approximate conversion: Agtron 135 ≈ 395-400°F, Agtron 60 ≈ 430-435°F
            # Linear interpolation
            if agtron_number >= 120:  # Very light
                return 395.0
            elif agtron_number >= 100:  # Light
                return 405.0
            elif agtron_number >= 80:  # Medium-light
                return 410.0
            elif agtron_number >= 60:  # Medium
                return 415.0
            elif agtron_number >= 50:  # Medium-dark
                return 420.0
            else:  # Dark
                return 425.0
        
        # Otherwise use text description
        if roast_level:
            level_lower = roast_level.lower()
            if 'light' in level_lower or 'expressive light' in level_lower:
                return 400.0
            elif 'medium-light' in level_lower:
                return 410.0
            elif 'medium' in level_lower:
                return 415.0
            elif 'dark' in level_lower or 'full city' in level_lower:
                return 425.0
        
        # Default to medium
        return 410.0
    
    def scrape_roast_profile(self, url):
        """
        Step 2: Scrape individual roast profile with enhanced metadata
        """
        driver = self.setup_driver()
        
        try:
            driver.get(url)
            time.sleep(8)  # Wait for amCharts to load
            
            # Extract enhanced metadata
            metadata = self.extract_enhanced_metadata(driver, url)
            
            # Check if chart exists
            try:
                chart_div = driver.find_element(By.ID, "chartdiv")
            except:
                print(f"    ⚠️  No chart found on page")
                return None
            
            # Extract roast profile from amCharts
            chart_data_script = """
            if (window.am5 && window.am5.registry.rootElements) {
                let roots = window.am5.registry.rootElements;
                
                if (roots.length > 0) {
                    let root = roots[0];
                    let chart = root.container.children.values[0];
                    
                    let seriesData = [];
                    
                    if (chart.series) {
                        chart.series.each(function(series, index) {
                            let data = series.data.values;
                            let seriesName = series.get("name") || `Series_${index}`;
                            let yAxis = series.get("yAxis");
                            let yAxisRenderer = yAxis ? yAxis.get("renderer") : null;
                            
                            seriesData.push({
                                name: seriesName,
                                opposite: yAxisRenderer ? yAxisRenderer.get("opposite") : false,
                                data: data,
                                series_index: index
                            });
                        });
                    }
                    
                    return {
                        found: true,
                        series: seriesData
                    };
                }
            }
            
            return {found: false, message: "amCharts not found"};
            """
            
            result = driver.execute_script(chart_data_script)
            
            if not result.get('found'):
                print(f"    ✗ No chart data found")
                return None
            
            # Parse series data
            profile = {
                'metadata': metadata,
                'roast_profile': {},
                'scraped_at': datetime.now().isoformat(),
                'method': 'amcharts_registry_v3'
            }
            
            for series_info in result['series']:
                series_name = series_info['name'].lower()
                data = series_info['data']
                opposite = series_info.get('opposite', False)
                
                # Map to standard names
                if 'bean' in series_name or 'temp' in series_name:
                    if not opposite:
                        profile['roast_profile']['bean_temp'] = data
                    else:
                        profile['roast_profile'][series_name.replace(' ', '_')] = data
                elif 'ror' in series_name or 'rate' in series_name:
                    profile['roast_profile']['rate_of_rise'] = data
                else:
                    if series_info['series_index'] == 0:
                        profile['roast_profile']['bean_temp'] = data
                    elif series_info['series_index'] == 1:
                        profile['roast_profile']['rate_of_rise'] = data
                    else:
                        profile['roast_profile'][f"series_{series_info['series_index']}"] = data
            
            # Calculate summary statistics
            if 'bean_temp' in profile['roast_profile']:
                temps = profile['roast_profile']['bean_temp']
                times = [p.get('time', 0) for p in temps]
                values = [p.get('value', 0) for p in temps]
                
                if times and values:
                    profile['summary'] = {
                        'duration_seconds': max(times),
                        'duration_minutes': max(times) / 60,
                        'data_points': len(temps),
                        'start_temp': values[0],
                        'finish_temp': values[-1],
                        'min_temp': min(values),
                        'max_temp': max(values),
                        'turning_point_temp': min(values),
                        'turning_point_time': times[values.index(min(values))],
                    }
            
            return profile
            
        except Exception as e:
            print(f"    ✗ Error scraping profile: {e}")
            return None
            
        finally:
            driver.quit()
    
    def build_complete_dataset(self, max_products=None, resume_from=0):
        """
        Step 3: Build complete dataset with enhanced features
        """
        print("\n" + "="*80)
        print("STEP 2: Scraping roast profiles with Phase 1 + Phase 2 features")
        print("="*80)
        
        # Load product URLs
        product_urls_file = f"{self.output_dir}/product_urls.json"
        
        if not os.path.exists(product_urls_file):
            print("⚠️  No product URLs found. Running discovery first...")
            product_urls = self.find_all_product_urls()
        else:
            with open(product_urls_file, 'r') as f:
                product_urls = json.load(f)
            print(f"✓ Loaded {len(product_urls)} product URLs from file")
        
        if not product_urls:
            print("✗ No products to scrape")
            return None
        
        # Apply limits
        if resume_from > 0:
            product_urls = product_urls[resume_from:]
            print(f"📝 Resuming from product #{resume_from + 1}")
        
        if max_products:
            product_urls = product_urls[:max_products]
            print(f"📝 Limiting to {max_products} products")
        
        print(f"\n🔄 Scraping {len(product_urls)} products...")
        print("="*80)
        
        successful_profiles = []
        failed_products = []
        
        for i, url in enumerate(product_urls, start=resume_from + 1):
            product_name = url.split('/')[-1].replace('-', ' ').title()
            
            print(f"\n[{i}/{resume_from + len(product_urls)}] {product_name}")
            print(f"  URL: {url}")
            
            profile = self.scrape_roast_profile(url)
            
            if profile:
                # Check if this is a new batch
                product_name = profile['metadata'].get('product_name')
                batch_num = profile['metadata'].get('roast_info', {}).get('batch')
                roast_date = profile['metadata'].get('roast_info', {}).get('roast_date')
                
                is_new, reason = self._check_if_new_batch(product_name, batch_num, roast_date)
                
                if not is_new:
                    print(f"  ⊘ Skipped: {reason}")
                    failed_products.append({
                        'index': i,
                        'name': product_name,
                        'url': url,
                        'skip_reason': reason
                    })
                    time.sleep(1)  # Small delay even for skipped
                    continue
                
                # Save individual profile with batch suffix
                safe_filename = re.sub(r'[^\w\-]', '_', url.split('/')[-1])
                
                # Add batch number to filename if available
                if batch_num:
                    profile_file = f"{self.output_dir}/profiles/{safe_filename}_batch{batch_num}.json"
                else:
                    # Use timestamp if no batch number
                    timestamp = datetime.now().strftime('%H%M%S')
                    profile_file = f"{self.output_dir}/profiles/{safe_filename}_{timestamp}.json"
                
                with open(profile_file, 'w') as f:
                    json.dump(profile, f, indent=2, default=str)
                
                successful_profiles.append(profile)
                print(f"  ✓ Saved: {os.path.basename(profile_file)}")
                print(f"  📝 {reason}")
                
                if 'summary' in profile:
                    s = profile['summary']
                    print(f"  📊 {s['duration_seconds']}s, "
                          f"{s['start_temp']:.0f}°F → {s['finish_temp']:.0f}°F")
            else:
                failed_products.append({
                    'index': i,
                    'name': product_name,
                    'url': url
                })
                print(f"  ✗ Failed")
            
            # Be nice to servers
            time.sleep(3)
            
            # Save progress every 10 products
            if i % 10 == 0:
                self._save_progress(successful_profiles, failed_products, i)
        
        # Save complete dataset
        dataset = {
            'dataset_info': {
                'created_at': datetime.now().isoformat(),
                'scrape_date': datetime.now().strftime('%Y-%m-%d'),
                'version': 'v3.1_additive',
                'features': 'Phase 1 + Phase 2 + Flavors (Origin, Process, Roast Level, Variety, Altitude, Drying, Flavor Notes)',
                'total_products_discovered': resume_from + len(product_urls),
                'total_attempted': len(product_urls),
                'successful_scrapes': len(successful_profiles),
                'failed_scrapes': len(failed_products),
                'skipped_existing_batches': len([f for f in failed_products if 'skip_reason' in f]),
                'success_rate': f"{len(successful_profiles)/len(product_urls)*100:.1f}%",
                'output_directory': self.output_dir
            },
            'profiles': successful_profiles,
            'failed_products': failed_products
        }
        
        dataset_file = f"{self.output_dir}/complete_dataset.json"
        with open(dataset_file, 'w') as f:
            json.dump(dataset, f, indent=2, default=str)
        
        print("\n" + "="*80)
        print("ADDITIVE DATASET BUILD COMPLETE")
        print("="*80)
        print(f"✓ New profiles: {len(successful_profiles)}/{len(product_urls)}")
        print(f"⊘ Skipped (existing): {len([f for f in failed_products if 'skip_reason' in f])}")
        print(f"✗ Failed: {len([f for f in failed_products if 'skip_reason' not in f])}")
        print(f"📈 New profile rate: {dataset['dataset_info']['success_rate']}")
        print(f"\n✓ Complete dataset: {dataset_file}")
        print(f"✓ Individual profiles: {self.output_dir}/profiles/")
        print(f"✓ Directory: {self.output_dir}/")
        
        # Print batch statistics
        total_historical = sum(len(batches) for batches in self.batch_history.values())
        print(f"\n📊 Historical tracking:")
        print(f"   Total products tracked: {len(self.batch_history)}")
        print(f"   Total batches in history: {total_historical}")
        print(f"   New batches this scrape: {len(successful_profiles)}")
        
        return dataset
    
    def _save_progress(self, successful_profiles, failed_products, current_index):
        """Save progress checkpoint"""
        checkpoint = {
            'checkpoint_at': datetime.now().isoformat(),
            'products_processed': current_index,
            'successful_count': len(successful_profiles),
            'failed_count': len(failed_products),
        }
        
        checkpoint_file = f"{self.output_dir}/logs/checkpoint_{current_index}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"  💾 Progress saved")
    
    def export_to_csv(self):
        """
        Export enhanced dataset to CSV with all Phase 1 + Phase 2 features
        """
        print("\n" + "="*80)
        print("STEP 3: Exporting enhanced dataset to CSV")
        print("="*80)
        
        dataset_file = f"{self.output_dir}/complete_dataset.json"
        
        if not os.path.exists(dataset_file):
            print("✗ No dataset found.")
            return None
        
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        
        rows = []
        
        for profile in dataset['profiles']:
            row = {
                # Basic info
                'product_name': profile['metadata'].get('product_name'),
                'url': profile['metadata'].get('url'),
                
                # Phase 1 features
                'origin': profile['metadata'].get('origin'),
                'process': profile['metadata'].get('process'),
                'roast_level': profile['metadata'].get('roast_level'),
                'roast_level_agtron': profile['metadata'].get('roast_level_agtron'),
                'target_finish_temp': profile['metadata'].get('target_finish_temp'),
                
                # Phase 2 features
                'variety': profile['metadata'].get('variety'),
                'altitude': profile['metadata'].get('altitude'),
                'altitude_numeric': profile['metadata'].get('altitude_numeric'),
                'bean_density_proxy': profile['metadata'].get('bean_density_proxy'),
                'drying_method': profile['metadata'].get('drying_method'),
                
                # Additional context
                'harvest_season': profile['metadata'].get('harvest_season'),
                'roaster_machine': profile['metadata'].get('roaster_machine'),
                'preferred_extraction': profile['metadata'].get('preferred_extraction'),
                'caffeine_mg': profile['metadata'].get('caffeine_mg'),
                
                # Flavor notes
                'flavor_notes_raw': profile['metadata'].get('flavor_notes_raw'),
                'flavor_notes_parsed': ', '.join(profile['metadata'].get('flavor_notes_parsed', [])) if profile['metadata'].get('flavor_notes_parsed') else None,
                'flavor_categories': ', '.join(profile['metadata'].get('flavor_categories', [])) if profile['metadata'].get('flavor_categories') else None,
            }
            
            # Add roast info
            if 'roast_info' in profile['metadata']:
                for key, value in profile['metadata']['roast_info'].items():
                    row[f'roast_{key}'] = value
            
            # Add summary stats
            if 'summary' in profile:
                row.update(profile['summary'])
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_file = f"{self.output_dir}/dataset_summary.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"✓ CSV exported: {csv_file}")
        print(f"✓ {len(df)} profiles × {len(df.columns)} columns")
        
        # Print feature statistics
        print(f"\n📊 Feature Coverage:")
        feature_cols = ['origin', 'process', 'roast_level', 'variety', 'altitude', 'drying_method', 'flavor_notes_raw']
        for col in feature_cols:
            if col in df.columns:
                coverage = (df[col].notna().sum() / len(df)) * 100
                print(f"  {col:20s}: {coverage:5.1f}% ({df[col].notna().sum()}/{len(df)})")
        
        # Print value ranges
        print(f"\n📈 Value Ranges:")
        if 'altitude_numeric' in df.columns:
            alt_df = df['altitude_numeric'].dropna()
            if len(alt_df) > 0:
                print(f"  Altitude: {alt_df.min():.0f}-{alt_df.max():.0f}m (mean: {alt_df.mean():.0f}m)")
        
        if 'target_finish_temp' in df.columns:
            temp_df = df['target_finish_temp'].dropna()
            if len(temp_df) > 0:
                print(f"  Target Finish Temp: {temp_df.min():.0f}-{temp_df.max():.0f}°F (mean: {temp_df.mean():.0f}°F)")
        
        return df
    
    def create_readme(self):
        """Create README with Phase 1 + Phase 2 documentation"""
        
        dataset_file = f"{self.output_dir}/complete_dataset.json"
        if os.path.exists(dataset_file):
            with open(dataset_file, 'r') as f:
                dataset = json.load(f)
            dataset_info = dataset.get('dataset_info', {})
        else:
            dataset_info = {}
        
        readme_content = f"""# Onyx Coffee Lab Roast Profile Dataset - ENHANCED v3.0

**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Source:** https://onyxcoffeelab.com  
**Purpose:** Validation dataset for RoastFormer with comprehensive bean characteristics  
**Features:** Phase 1 + Phase 2 conditioning variables

## Dataset Overview

- **Total Profiles:** {dataset_info.get('successful_scrapes', 'N/A')}
- **Success Rate:** {dataset_info.get('success_rate', 'N/A')}
- **Version:** {dataset_info.get('version', 'v3.0_enhanced')}

## Enhanced Features

### Phase 1: Critical Conditioning Variables
1. **Origin** - Geographic region (Ethiopia, Colombia, Kenya, etc.)
2. **Process** - Processing method (Washed, Natural, Honey, Anaerobic)
3. **Roast Level** - Target roast description (Expressive Light, Medium, Dark)
4. **Roast Level Agtron** - Numeric Agtron value (e.g., #135)
5. **Target Finish Temp** - Inferred target finish temperature (°F)

### Phase 2: Helpful Features
6. **Variety** - Coffee variety (Mixed, Heirloom, Caturra, Bourbon, Geisha)
7. **Altitude** - Growing altitude (e.g., "1500 MASL")
8. **Altitude Numeric** - Numeric altitude in meters
9. **Bean Density Proxy** - Calculated from altitude (g/cm³)
10. **Drying Method** - Post-harvest drying (Raised-Bed, Patio, etc.)

### Phase 3: Flavor Profile Features (NEW!)
11. **Flavor Notes Raw** - Raw flavor text ("BERRIES STONE FRUIT EARL GREY HONEYSUCKLE ROUND")
12. **Flavor Notes Parsed** - Individual flavor descriptors as list
13. **Flavor Categories** - Categorized flavors (fruity, floral, chocolate, etc.)

### Additional Context
11. **Harvest Season** - When harvested (Rotating Microlots, October, etc.)
12. **Roaster Machine** - Production roaster (Loring S70 Peregrine, etc.)
13. **Preferred Extraction** - Intended use (Filter, Espresso, Both)
14. **Caffeine Content** - Caffeine in mg per 12oz cup

## Transformer Conditioning Usage

### Categorical Features (Embedding Layers)
```python
# Origin: {{'Ethiopia': 0, 'Colombia': 1, 'Kenya': 2, ...}}
# Process: {{'Washed': 0, 'Natural': 1, 'Honey': 2, 'Anaerobic': 3}}
# Roast Level: {{'Light': 0, 'Medium': 1, 'Dark': 2}}
# Variety: {{'Mixed': 0, 'Heirloom': 1, 'Caturra': 2, ...}}
```

### Continuous Features (Direct Input)
```python
# target_finish_temp: 395-425°F (normalized 0-1)
# altitude_numeric: 1000-2500m (normalized 0-1)
# bean_density_proxy: 0.65-0.80 g/cm³ (normalized 0-1)
# caffeine_mg: 180-230mg (normalized 0-1)
```

### Example Conditioning Code
```python
import torch
import torch.nn as nn

# Embedding layers for categorical features
origin_embed = nn.Embedding(num_origins, embed_dim)
process_embed = nn.Embedding(num_processes, embed_dim)
roast_level_embed = nn.Embedding(num_roast_levels, embed_dim)
variety_embed = nn.Embedding(num_varieties, embed_dim)

# Flavor embeddings (average multiple flavor notes)
flavor_embed = nn.Embedding(num_flavors, embed_dim)

# Continuous feature projection
continuous_features = torch.tensor([
    target_finish_temp / 425.0,  # Normalize to 0-1
    altitude_numeric / 2500.0,
    bean_density_proxy / 0.80,
])
continuous_proj = nn.Linear(3, embed_dim)(continuous_features)

# Flavor encoding (average embeddings for multiple flavors)
flavor_indices = [flavor_vocab['berries'], flavor_vocab['floral'], flavor_vocab['citrus']]
flavor_embeds = [flavor_embed(torch.tensor(idx)) for idx in flavor_indices]
avg_flavor_embed = torch.mean(torch.stack(flavor_embeds), dim=0)

# Combine all conditioning
condition_vector = torch.cat([
    origin_embed(origin_idx),
    process_embed(process_idx),
    roast_level_embed(roast_level_idx),
    variety_embed(variety_idx),
    avg_flavor_embed,  # NEW: Flavor conditioning!
    continuous_proj
], dim=-1)

# Feed to transformer
output = transformer(temperature_sequence, condition=condition_vector)
```

## Directory Structure

```
{self.output_dir}/
├── complete_dataset.json      # Full dataset with all features
├── dataset_summary.csv         # CSV with all Phase 1 + Phase 2 features
├── product_urls.json           # Discovered product URLs
├── profiles/                   # Individual profile JSONs
├── logs/                       # Progress checkpoints
└── README.md                   # This file
```

## Feature Coverage

Expected coverage rates:
- **Origin:** ~95-100% (almost always present)
- **Process:** ~80-90% (common but not always listed)
- **Roast Level:** ~100% (always present)
- **Variety:** ~70-90% (varies by coffee)
- **Altitude:** ~60-80% (not always specified)
- **Drying Method:** ~40-60% (less commonly detailed)

## Bean Density Calculation

Bean density proxy is estimated from altitude:
```python
# Formula: density = 0.65 + (altitude_km * 0.05)
# Example: 2000m altitude → 0.65 + (2.0 * 0.05) = 0.75 g/cm³
# 
# Typical ranges:
# - Low altitude (< 1000m): 0.65-0.70 g/cm³
# - Mid altitude (1000-1800m): 0.70-0.74 g/cm³
# - High altitude (1800-2500m): 0.74-0.78 g/cm³
```

## Roast Level to Temperature Mapping

Target finish temperatures inferred from Agtron values:
- **Agtron 120+** (Very Light): 395°F
- **Agtron 100-119** (Light): 405°F
- **Agtron 80-99** (Medium-Light): 410°F
- **Agtron 60-79** (Medium): 415°F
- **Agtron 50-59** (Medium-Dark): 420°F
- **Agtron < 50** (Dark): 425°F

## Citation

```
Onyx Coffee Lab Enhanced Roast Profile Dataset v3.0
Source: https://onyxcoffeelab.com
Accessed: {datetime.now().strftime('%B %Y')}
Features: Phase 1 + Phase 2 conditioning variables for transformer training
Purpose: RoastFormer validation dataset
```

## Version History

- **v3.0** ({datetime.now().strftime('%Y-%m-%d')}): Enhanced feature extraction
  - Added Phase 1 features: Origin, Process, Roast Level, Target Finish Temp
  - Added Phase 2 features: Variety, Altitude, Bean Density, Drying Method
  - Added contextual features: Harvest Season, Roaster Machine, Extraction Method
  - {dataset_info.get('successful_scrapes', 'N/A')} roast profiles

---

**Ready for RoastFormer Transformer Training! ☕🤖**
"""
        
        readme_file = f"{self.output_dir}/README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        print(f"✓ Enhanced README created: {readme_file}")
        return readme_file


def main():
    """Main execution - ADDITIVE mode with date-stamped directories"""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║       ONYX DATASET BUILDER v3.1 ADDITIVE                   ║
    ║       Date-Stamped Directories + Batch Tracking            ║
    ║       For RoastFormer Transformer Conditioning             ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize builder (automatically creates date-stamped directory)
    builder = OnyxDatasetBuilderV3()
    
    print("\n📅 ADDITIVE MODE:")
    print("  ✓ Date-stamped directory created automatically")
    print("  ✓ Checks all previous scrapes for batch history")
    print("  ✓ Only scrapes NEW batches (won't duplicate)")
    print("  ✓ Safe to run multiple times!")
    
    # Step 1: Discover products
    product_urls = builder.find_all_product_urls()
    
    if not product_urls:
        print("\n✗ No products discovered. Exiting.")
        return
    
    print(f"\n✓ Discovered {len(product_urls)} products")
    
    # Step 2: Build dataset (default to full)
    dataset = builder.build_complete_dataset(max_products=None)
    
    if not dataset or not dataset['profiles']:
        print("\n⚠️  No NEW profiles scraped.")
        print("    All current batches already in previous scrapes!")
        print("    Try again in a few days when Onyx updates.")
        return
    
    # Step 3: Export to CSV
    df = builder.export_to_csv()
    
    # Step 4: Create documentation
    builder.create_readme()
    
    print("\n" + "="*80)
    print("🎉 ADDITIVE SCRAPE COMPLETE!")
    print("="*80)
    print(f"\n📁 This scrape: {builder.output_dir}/")
    print(f"   ✓ {len(dataset['profiles'])} NEW profiles")
    
    skipped = len([f for f in dataset['failed_products'] if 'skip_reason' in f])
    if skipped > 0:
        print(f"   ⊘ {skipped} skipped (already have those batches)")
    
    print("\n📊 Features extracted:")
    print("   Phase 1: Origin, Process, Roast Level, Target Finish Temp")
    print("   Phase 2: Variety, Altitude, Bean Density, Drying Method")
    print("   Phase 3: Flavor Notes, Flavor Categories")
    
    print("\n💡 Run again in a few days to collect new batches!")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            builder = OnyxDatasetBuilderV3()
            builder.find_all_product_urls()
            builder.build_complete_dataset(max_products=3)
            builder.export_to_csv()
            builder.create_readme()
        elif sys.argv[1] == "--resume":
            resume_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            builder = OnyxDatasetBuilderV3()
            builder.build_complete_dataset(resume_from=resume_index)
            builder.export_to_csv()
            builder.create_readme()
    else:
        main()

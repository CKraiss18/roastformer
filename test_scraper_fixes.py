"""
Quick test script to verify Onyx scraper fixes
Tests the fixed scraper on a single product and shows what was extracted
"""

import sys
import json
from onyx_dataset_builder_v3_3_COMBINED import OnyxDatasetBuilderV3

def test_single_product(url):
    """
    Test scraping a single product and show detailed extraction results
    """
    print("="*80)
    print("TESTING ONYX SCRAPER FIXES")
    print("="*80)
    print(f"\nTest URL: {url}")
    print("\n" + "-"*80)
    
    # Initialize builder (use_date_suffix=False for testing)
    builder = OnyxDatasetBuilderV3(output_dir='test_scrape', use_date_suffix=False)
    
    # Scrape the product
    print("\n📡 Scraping product...")
    result = builder.scrape_roast_profile(url)
    
    if not result:
        print("\n❌ SCRAPING FAILED")
        return
    
    metadata = result.get('metadata', {})
    
    print("\n" + "="*80)
    print("EXTRACTION RESULTS")
    print("="*80)
    
    # Check critical fields
    print("\n🏷️  PRODUCT NAME:")
    product_name = metadata.get('product_name', 'NOT FOUND')
    print(f"   Value: '{product_name}'")
    
    # Check for character spacing issue
    if ' ' in product_name and len(product_name.replace(' ', '')) < len(product_name) / 2:
        print("   ⚠️  WARNING: Product name still has character spacing!")
    else:
        print("   ✅ OK - No character spacing issues")
    
    print("\n🌍 ORIGIN:")
    origin = metadata.get('origin', 'NOT FOUND')
    print(f"   Value: '{origin}'")
    print(f"   Status: {'✅ FOUND' if origin != 'NOT FOUND' else '❌ MISSING'}")
    
    print("\n⚗️  PROCESS METHOD:")
    process = metadata.get('process', 'NOT FOUND')
    print(f"   Value: '{process}'")
    print(f"   Status: {'✅ FOUND' if process != 'NOT FOUND' else '❌ MISSING'}")
    
    print("\n☕ ROAST LEVEL:")
    roast = metadata.get('roast_level', 'NOT FOUND')
    agtron = metadata.get('roast_level_agtron', 'N/A')
    print(f"   Level: '{roast}'")
    print(f"   Agtron: {agtron}")
    print(f"   Status: {'✅ FOUND' if roast != 'NOT FOUND' else '❌ MISSING'}")
    
    print("\n🎨 FLAVOR NOTES:")
    flavor_raw = metadata.get('flavor_notes_raw')
    flavor_parsed = metadata.get('flavor_notes_parsed')
    flavor_cats = metadata.get('flavor_categories')
    
    if flavor_raw:
        print(f"   Raw: '{flavor_raw}'")
        print(f"   Parsed: {flavor_parsed}")
        print(f"   Categories: {flavor_cats}")
        print("   ✅ FLAVORS EXTRACTED!")
    else:
        print("   ❌ NO FLAVORS FOUND")
    
    print("\n🌾 VARIETY:")
    variety = metadata.get('variety', 'NOT FOUND')
    print(f"   Value: '{variety}'")
    print(f"   Status: {'✅ FOUND' if variety != 'NOT FOUND' else '❌ MISSING'}")
    
    print("\n⛰️  ALTITUDE:")
    altitude = metadata.get('altitude', 'NOT FOUND')
    altitude_num = metadata.get('altitude_numeric')
    print(f"   Value: '{altitude}'")
    if altitude_num:
        print(f"   Numeric: {altitude_num}m")
    print(f"   Status: {'✅ FOUND' if altitude != 'NOT FOUND' else '❌ MISSING'}")
    
    print("\n📊 ROAST PROFILE:")
    roast_profile = result.get('roast_profile', {})
    if 'bean_temp' in roast_profile:
        bean_data = roast_profile['bean_temp']
        print(f"   Bean Temp: {len(bean_data)} points")
        if bean_data:
            print(f"   Start: {bean_data[0].get('value', 'N/A')}°F at {bean_data[0].get('time', 'N/A')}s")
            print(f"   End: {bean_data[-1].get('value', 'N/A')}°F at {bean_data[-1].get('time', 'N/A')}s")
            
            # Show first 10 and last 10 points for verification
            print(f"\n   📈 First 10 data points:")
            for point in bean_data[:10]:
                print(f"      Time {point.get('time')}s: {point.get('value')}°F")
            
            print(f"\n   📈 Last 10 data points:")
            for point in bean_data[-10:]:
                print(f"      Time {point.get('time')}s: {point.get('value')}°F")
            
        print("   ✅ PROFILE DATA EXTRACTED")
    else:
        print("   ❌ NO PROFILE DATA FOUND")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    critical_fields = {
        'Product Name': product_name != 'NOT FOUND',
        'Origin': origin != 'NOT FOUND',
        'Process': process != 'NOT FOUND',
        'Flavors': flavor_raw is not None,
        'Roast Level': roast != 'NOT FOUND',
        'Profile Data': 'bean_temp' in roast_profile
    }
    
    found_count = sum(critical_fields.values())
    total_count = len(critical_fields)
    
    print(f"\nCritical Fields Found: {found_count}/{total_count}")
    
    for field, found in critical_fields.items():
        status = "✅" if found else "❌"
        print(f"  {status} {field}")
    
    if found_count == total_count:
        print("\n🎉 ALL CRITICAL FIELDS EXTRACTED SUCCESSFULLY!")
    elif found_count >= total_count * 0.7:
        print("\n⚠️  Most fields extracted, but some are missing")
    else:
        print("\n❌ SCRAPER NEEDS MORE FIXES")
    
    # Save detailed results INCLUDING FULL ROAST PROFILE
    output_file = "test_scrape/test_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'url': url,
            'metadata': metadata,
            'roast_profile': result.get('roast_profile', {}),  # FULL profile data
            'summary': result.get('summary', {}),  # Profile summary stats
            'critical_fields': critical_fields
        }, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    # Default test URL (Geometry)
    test_url = "https://onyxcoffeelab.com/products/geometry?variant=31862717677666"
    
    # Allow custom URL from command line
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
    
    test_single_product(test_url)
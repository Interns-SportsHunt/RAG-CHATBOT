import os
import json
import requests
from bs4 import BeautifulSoup
import re
import time
import argparse

# =========================
# ASCII Art Banners
# =========================
BANNER = r"""
    _____                                  
  / ____|                                 
 | (___   ___ _ __ __ _ _ __   ___ _ __   
  \___ \ / __| '__/ _` | '_ \ / _ \ '__|  
  ____) | (__| | | (_| | |_) |  __/ |     
 |_____/ \___|_|  \__,_| .__/ \___|_|     
                       | |                
                       |_|                
"""

FOOTER = r"""
===============================================================
      All course data has been scraped and saved! 
===============================================================
"""

SUCCESS = "[✔]"
FAIL =    "[✗]"

# =========================
# Helper Functions
# =========================

def slugify(value):
    """
    Converts a string to a slugified version suitable for filenames.
    Replaces non-alphanumeric characters with underscores and lowercases the result.
    """
    value = str(value)
    value = re.sub(r'[\W_]+', '_', value)
    return value.strip('_').lower()

# =========================
# Setup Output Directory
# =========================
curr_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(curr_dir, '..', 'data', 'courses_json')
os.makedirs(target_dir, exist_ok=True)
output_dir = target_dir

# =========================
# User Interface
# =========================

print(BANNER)
print("Welcome to the Course Scraper!")
print("===============================================================\n")

# =========================
# Load Input Data
# =========================

parser = argparse.ArgumentParser(description="Scrape course data from a JSON file.")
parser.add_argument('--input', '-i', required=True, help='Path to the input JSON file containing courses data')
args = parser.parse_args()

with open(args.input, 'r', encoding='utf-8') as f:
    courses = json.load(f)

total = len(courses)
success_count = 0
fail_count = 0
failed_courses = [] 

# =========================
# Main Scraping Loop
# =========================

for idx, course in enumerate(courses, 1):
    name = course.get('name')
    urls = course.get('url', [])
    locations = course.get('locations', [])

    print(f"\n{'='*60}")
    print(f"Course {idx}/{total}")
    print(f"Name      : {name}")
    print(f"Location  : {', '.join(locations) if locations else 'N/A'}")

    # Skip courses with no URL
    if not urls:
        print(f"{FAIL} No URL for course: {name}")
        fail_count += 1
        failed_courses.append(course)
        continue

    url = urls[0]
    print(f"Fetching data from: {url}")

    # Fetch course page
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        print(f"{FAIL} Failed to fetch {url} for {name}: {e}")
        fail_count += 1
        failed_courses.append(course)
        continue

    soup = BeautifulSoup(html, 'html.parser')

    # -------------------------
    # Extract Course Information
    # -------------------------

    # 1. Course Title
    course_title = soup.find('h1', class_='course-hdd-txtds')
    course_title = course_title.get_text(strip=True) if course_title else None

    # 2. Department/School Name (from breadcrumbs)
    breadcrumbs = soup.select('section#brd-crmp-depp ul.no-pad li')
    department = breadcrumbs[1].get_text(strip=True) if len(breadcrumbs) > 1 else None

    # 3. Campus Name (from input data)
    campus = locations
    
    # 4. Course Status (Open/Closed)
    status_btn = soup.find('button', class_='courseApplyButtons')
    status = status_btn.get_text(strip=True) if status_btn else None

    # 5 & 6. Syllabus and Course Structure Links
    syllabus_link = None
    course_structure_link = None
    for a in soup.select('a[href]'):
        if 'syllabus' in a['href']:
            syllabus_link = a['href']
        if 'syllabusstructure' in a['href']:
            course_structure_link = a['href']
    
    # 7. Fee Structure (tab content)
    fee_structure = None
    fee_structure_table = []
    fee_tab = soup.find('div', {'id': 'vtab_1'})
    if fee_tab:
        fee_table = fee_tab.find('table')
        if fee_table:
            headers = []
            rows = fee_table.find_all('tr')
            if rows:
                # Extract table headers
                ths = rows[0].find_all(['th', 'td'])
                headers = [th.get_text(strip=True) for th in ths]
                # Extract each row as a dictionary
                for row in rows[1:]:
                    cells = row.find_all(['th', 'td'])
                    if len(cells) == len(headers):
                        entry = {headers[i].lower().replace(' ', '_'): cells[i].get_text(strip=True) for i in range(len(headers))}
                        fee_structure_table.append(entry)
            if fee_structure_table:
                fee_structure = fee_structure_table
            else:
                fee_structure = fee_table.get_text(' ', strip=True)

    # 8. Eligibility Criteria (tab content)
    eligibility = None
    elig_tab = soup.find('div', {'id': 'vtab_0'})
    if elig_tab:
        # Only extract the actual eligibility text, not PEO/PO/PSO
        paragraphs = elig_tab.find_all('p')
        eligibility_lines = []
        for p in paragraphs:
            txt = p.get_text(' ', strip=True)
            if txt:
                eligibility_lines.append(txt)
        # Filter to only keep lines starting from 'Eligibility for the programme' and including 'Lateral Entry' if present
        elig_text = '\n'.join(eligibility_lines)
        match = re.search(r'(Eligibility for the programme.*?)(Lateral Entry:.*?programmes\.)', elig_text, re.DOTALL)
        if match:
            eligibility = match.group(1) + '\n' + match.group(2)
        else:
            # fallback: just keep lines containing 'Eligibility' or 'Lateral Entry'
            filtered = []
            found_elig = False
            for line in eligibility_lines:
                if 'Eligibility for the programme' in line:
                    found_elig = True
                if found_elig:
                    filtered.append(line)
                if 'Lateral Entry' in line:
                    break
            if filtered:
                eligibility = '\n'.join(filtered)
            else:
                eligibility = elig_text

    # 9. Why Choose This Course? (section id="why-choos-bck-foff")
    why_choose = None
    why_section = soup.find('section', {'id': 'why-choos-bck-foff'})
    if why_section:
        why_choose = why_section.get_text(' ', strip=True)

    # 10. What You Will Learn (section id="u-will-lern-bk-off")
    what_learn = None
    learn_section = soup.find('section', {'id': 'u-will-lern-bk-off'})
    if learn_section:
        what_learn = learn_section.get_text(' ', strip=True)

    # 11. Modules/Curriculum Structure (section id="modules-321")
    modules = None
    modules_section = soup.find('section', {'id': 'modules-321'})
    if modules_section:
        modules = modules_section.get_text(' ', strip=True)

    # 12. Career Prospects (section id="carees-prosp")
    career = None
    career_section = soup.find('section', {'id': 'carees-prosp'})
    if career_section:
        career = career_section.get_text(' ', strip=True)

    # 13. Admission Process (tab vtabCOM_5)
    admission_process = None
    adm_tab = soup.find('div', {'id': 'vtabCOM_5'})
    if adm_tab:
        # Extract all text in order, preserving structure
        parts = []
        for tag in adm_tab.find_all(['div', 'ol', 'ul', 'p'], recursive=True):
            text = tag.get_text(' ', strip=True)
            if text:
                parts.append(text)
        admission_process = '\n'.join(parts)
        # Clean up: Only keep from 'Results and Admission Process' to 'Cancellation Process' (if present)
        match = re.search(
            r'(Results and Admission Process.*?)(Cancellation Process:.*?)(For details.*?cancellation1)',
            admission_process,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            admission_process = match.group(1) + '\n' + match.group(2) + '\n' + match.group(3)
        else:
            # fallback: just keep from 'Results and Admission Process' onwards
            idx = admission_process.lower().find('results and admission process')
            if idx != -1:
                admission_process = admission_process[idx:]

    # -------------------------
    # Merge and Save Data
    # -------------------------

    data = {
        'course_title': course_title,
        'department': department,
        'campus': campus,
        'syllabus_link': syllabus_link,
        'course_structure_link': course_structure_link,
        'fee_structure': fee_structure,
        'eligibility': eligibility,
        'why_choose': why_choose,
        'what_learn': what_learn,
        'modules': modules,
        'career': career,
        'source_url': url
    }

    # Save the extracted data as a JSON file named after the slugified course name
    out_path = os.path.join(output_dir, f"{slugify(name)}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"{SUCCESS} Saved: {out_path}")
    success_count += 1
    time.sleep(0.2)  # Small delay for user experience

# =========================
# Handle Failed Courses
# =========================

if failed_courses:
    with open('failed_courses.json', 'w', encoding='utf-8') as f:
        json.dump(failed_courses, f, ensure_ascii=False, indent=2)
    print(f"\nA list of failed courses has been saved to failed_courses.json\n")
else:
    print("\nAll courses were scraped successfully! No failed_courses.json generated.\n")

# =========================
# Final Summary
# =========================

print("\n" + FOOTER)
print(f"Total courses processed: {total}")
print(f"Successfully scraped   : {success_count}")
print(f"Failed                : {fail_count}")
print("Thank you")

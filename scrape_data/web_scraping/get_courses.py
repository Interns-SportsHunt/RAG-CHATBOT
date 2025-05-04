import requests
from bs4 import BeautifulSoup
import json
import os

base_url = "https://christuniversity.in/all-courses"
res = requests.get(base_url, headers={"User-Agent": "Mozilla/5.0"})
res.raise_for_status()

soup = BeautifulSoup(res.text, "html.parser")

# Extract course names from h5, URLs and their locations  
courses = []
for res in soup.select('.reslt-txt-detls ul.no-pad'):
    h5 = res.find('h5')
    if not h5:
        continue
    course_name = h5.get_text(strip=True)
    a_tags = res.find_all('a', href=lambda href: href and '/courses/' in href)
    urls = [a['href'] for a in a_tags if a]
    if not urls:
        continue
    locations = [a.get_text(strip=True) for a in a_tags if a]
    courses.append({'name': course_name, 'url': urls, 'locations': locations})

curr_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(curr_dir, '..', 'data')
os.makedirs(target_dir, exist_ok=True)
output_path = os.path.join(target_dir, 'all_courses.json')

with open(output_path, "w") as f:
    json.dump(courses, f, indent=4)
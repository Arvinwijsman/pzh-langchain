from bs4 import BeautifulSoup
from urllib.request import urlopen

web = "https://koop.gitlab.io/STOP/standaard/2.0.0-rc/regeling.html"
page = urlopen(web)
html = page.read().decode("utf-8")
soup = BeautifulSoup(html, "html.parser")

file_name = "./stop/output.txt"  # output file name
p = soup.get_text()

file = open(file_name, "w", encoding="utf-8")
file.write(p)
file.close()

print("File saved in", file_name)

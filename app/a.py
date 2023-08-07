import csv

data = [
    {"Serial No": 1, "Title": "Facebook", "Content": "Social networking platform", "Country Origin": "United States"},
    {"Serial No": 2, "Title": "Twitter", "Content": "Microblogging platform", "Country Origin": "United States"},
    {"Serial No": 3, "Title": "Instagram", "Content": "Photo and video sharing platform", "Country Origin": "United States"},
    {"Serial No": 4, "Title": "LinkedIn", "Content": "Professional networking platform", "Country Origin": "United States"},
    {"Serial No": 5, "Title": "YouTube", "Content": "Video-sharing platform", "Country Origin": "United States"},
    {"Serial No": 6, "Title": "TikTok", "Content": "Short-form video platform", "Country Origin": "China"},
    {"Serial No": 7, "Title": "WhatsApp", "Content": "Messaging platform", "Country Origin": "United States"},
    {"Serial No": 8, "Title": "Snapchat", "Content": "Multimedia messaging app", "Country Origin": "United States"},
    {"Serial No": 9, "Title": "WeChat", "Content": "Social media and messaging app", "Country Origin": "China"},
    {"Serial No": 10, "Title": "Reddit", "Content": "Social news aggregation platform", "Country Origin": "United States"},
]

# Define the CSV file path
csv_file = "social_media.csv"

# Write data to the CSV file
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    fieldnames = ["Serial No", "Title", "Content", "Country Origin"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    # Write the header row
    writer.writeheader()

    # Write the data rows
    writer.writerows(data)

print("CSV file created successfully.")

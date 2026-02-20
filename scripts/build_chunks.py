import os # used to walk through files
import re # used for regular expression
import json # used to convert md into json
from pathlib import Path

BASE_DIR = Path("knowledge_base")
OUTPUT_FILE = "chunks.json"

MIN_CHARS = 100 # min chars in a chunk, prevent too short & meaningless chunks

def read_markdown(file_path):
  with open(file_path, "r", encoding="utf-8") as f:
    return f.read()

# for resume spliting
def split_by_headers(text):
  '''
split markdown text based on their headers hierarchy (# , ## , ### )
keep header to its content
  '''
  sections = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE) # ?= :lookahead, keep header to its content
  return [s.strip() for s in sections if s.strip()]              # ^ : since flags=re.MULTILINE, ^ means beginning of every row not entire string
                                                                 # #{1,3} : # will appear 1~3 times
                                                                 # s.strip() will remove the blankspace at the end/ in front of s

# for interview Q&A
def split_interview_sections(text):
  '''
Split interview bank by ### headers.
Each ### section becomes one chunk.
  '''
  sections = re.split(r"(?=^### )", text, flags=re.MULTILINE)   # after read those interview question banks, every Q&A start with ### titles
  cleaned_sections = []
  for sec in sections:
    sec = sec.strip()
    if sec.startswith("###"):
      cleaned_sections.append(sec)
  return cleaned_sections

# build chunks based on text + metadata
def create_chunk(text, source_type, file_name, extra_meta=None):
  if len(text) < MIN_CHARS:
    return None
  metadata = {
    "source_type": source_type,
    "file_name": file_name,
  }
  if extra_meta:
    metadata.update(extra_meta)
  return {
    "content":text,
    "metadata":metadata}

# process md files
def process_file(file_path):
    text = read_markdown(file_path)
    file_name = file_path.name
    folder_name = file_path.parent.name
    chunks = []
    # Interview Bank
    if folder_name == "interview_bank":
        sections = split_interview_sections(text)
        for section in sections:
            header_match = re.match(r"^### (.*)", section)  # must start with ### and then (.*) will capture all the question contents after ###
            question_title = header_match.group(1) if header_match else "unknown" # if there is no match in capture group () then 'unknown'
            chunk = create_chunk(                                                 # group(0): all content that matched
                section,                                                          # group(1): the content in first capture group
                source_type="interview",
                file_name=file_name,
                extra_meta={"question": question_title}
            )
            if chunk:
                chunks.append(chunk)
    # Projects
    elif folder_name == "projects":
        sections = split_by_headers(text)
        for section in sections:
            header_match = re.match(r"^(#{1,3}) (.*)", section)
            section_title = header_match.group(2) if header_match else "unknown"  # first capture group is (#{1,3})
            chunk = create_chunk(                                                 # so second capture group is (.*)
                section,
                source_type="project",
                file_name=file_name,
                extra_meta={"section": section_title}
            )
            if chunk:
                chunks.append(chunk)
    # Resume
    elif folder_name == "resume":
        sections = split_by_headers(text)
        for section in sections:
            header_match = re.match(r"^(#{1,3}) (.*)", section)
            section_title = header_match.group(2) if header_match else "unknown"
            chunk = create_chunk(
                section,
                source_type="resume",
                file_name=file_name,
                extra_meta={"section": section_title}
            )
            if chunk:
                chunks.append(chunk)
    return chunks

# main code
def main():
    all_chunks = []

    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith(".md"):
                file_path = Path(root) / file
                file_chunks = process_file(file_path)
                all_chunks.extend(file_chunks)  # use extend instead of append

    print("=" * 50)
    print(f"Total chunks created: {len(all_chunks)}")
    print("=" * 50)

    type_counts = {}
    for chunk in all_chunks:
        t = chunk["metadata"]["source_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print("Chunk distribution:")
    for k, v in type_counts.items():
        print(f"  {k}: {v}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

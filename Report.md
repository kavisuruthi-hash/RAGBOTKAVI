1. Added 3 FAQ pdf's into data folder.
Modified the chunk_size from 1000 to 600 because for FAQ related pdfs, each question answer will go into one chunk instead of splitting into multiple chunks. so 50 questions means 50 chunks will be created. Had 57 questions overall in my data folder and got 50 chunks out of it.
Modified the chunk_overlap from 200 to 100 as chunk_size is also reduced to 600.
Modified TOP_K to 3 for more focused asnwers.
Modified the system prompt as per requirement and added follow up question as well.
Attached the test result doc with screenshots of testresult.
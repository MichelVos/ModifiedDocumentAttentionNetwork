#!/bin/bash

input_file="searchforfonts.txt"
output_file="font_urls.txt"

while IFS= read -r font_name; do
    search_query="${font_name%.ttf} ttf file site:fontsquirrel.com OR site:dafont.com OR site:github.com"
    
    echo "Searching for: $search_query"

    # Use DuckDuckGo HTML search (lightweight and scriptable)
    url=$(curl -sG "https://html.duckduckgo.com/html/" \
        --data-urlencode "q=$search_query" \
        | grep -Eo 'href="https?://[^"]+"' \
        | grep -Eo 'https?://[^"]+' \
        | head -n 1)

    if [[ -n "$url" ]]; then
        echo "$font_name -> $url" | tee -a "$output_file"
    else
        echo "$font_name -> [No URL found]" | tee -a "$output_file"
    fi
done < "$input_file"

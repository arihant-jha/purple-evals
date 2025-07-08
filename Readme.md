Inspired by Hamel's virtual cycle presented by Hamel in this talk at AI Engineer Convference

https://www.youtube.com/watch?v=eLXF0VojuSs

![virtuous-cycle](image.png)

I am going to follow this approach for a simple workflow
- Previously user mentions their cat's names ranked by favouriteness
- This time they mention dog's names ranked in reverse favouriteness
- This time query is to find the name of their nth favourite cat and mth favourite dog combination
- AI returns the name of the cat and dog

Requirements
1. Ability to see the prompt | llm_response | test_result(non-llm)
2. Test with llm_as a judge with accuracy as the metric


Process
1. I intentionally run it on a very small mode, `gemini-2.5-flash-lite`, and we get `16%` accuracy with it.
2. Common mistake that the model makes is it ignores the `least` text and even when I ask it go give me 2nd least favourite dog, returns me 2nd favourite dog. In `50%` cases where it is correct it is because the number is `3rd` for which least and most favourite are same.
3. Viewing the results in a viewable format.
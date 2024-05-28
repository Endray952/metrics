import openai


default_one_prompt = {"role": "system",
                 "content": "You are multi label classifier of mobile app reviews."
                            "The app is a utility for downloading additional content (maps, mods, seeds, skins, add-ons) for the game Minecraft. "
                            "Possible labels are: "
                            "1 - Positive review. The user explicitly praises the app. Examples: 'Very great and will be used a lot in the future 🥰', 'nice app!', 'great'"
                            "2 - Negative review. The user explicitly expresses dissatisfaction with the app. Examples: 'Shit doesn`t work', 'I give it 1 star because it is not enough to update, please fix that and I give you 5 stars from me, thank you' "
                            "3 - User suggest app improvement. The user provides suggestions on new content or functionality, asking for adding some new content or complains about lack of content without describing errors. Examples: 'The game is very good apart from one thing, I think there should be more options for things', 'it is a perfect program but it must support the older versions', 'They should add more mods, everything else is fine'"
                            "4 - General problem/error: Includes any mention of errors or problems. Examples: 'It's very nice but there are shortcomings', 'Mods and buildings don't work even maps so 1 star', 'Nothing happens😈'"
                            "5 - Advertisement complaints: Specifically about excessive ads. Examples: 'The application is very wonderful, but there are too many ads', 'Best mod for Minecraft but the ads was too long 😭 I'm gonna give it a 5stars if the ads get even shorter'"
                            "6 - Network problems/errors: Issues related to network errors, but not error in downloading content. Examples: 'Everything is good but slow downloading is a problem', 'ok, but when I go to the online game, it asks for the internet and some things are not being downloaded'"
                            "7 - Content complaints: General complaints about the app's content. Examples: 'All the best worlds have disappeared from you and pickaxes, don’t download it, this is evil', 'I like it but some don't work'"
                            "8 - Content does not work: Downloaded content does not work in Minecraft. Examples: 'It's a good app, but some cards don't work there.', '	Some add-ons don't work, but they're good.'"
                            "9 - Download error: Errors occurred while downloading content. Examples: 'This app is very good but the maps sometimes do not show and install', '	The best mod installer, although mods may not install and you will need to download other 4 stars'"
                            "10 - Functionality problems: Problems with the app's functionality, excluding content or network issues or issues with downloading content. Examples: 'It won't let me in it stays stuck loading', 'I clicked on koins and it doesn't give koinss, I'll try again.'"
                            "11 - Complaints about paid content: Issues with costs or the pricing policy. Examples: 'Very bad, this thing about taking currency is too horrible, you can't believe that someone has the nerve to do something like that, they can only benefit from it VERY BAD', 'donation trash app, there are no good mods at all, even for 59 rubles and constant advertising'"
                            "Put 0 if there are no suitable labels. Example: 'Hi!', 'Group', 'I ate coca-cola'"
                            "Choose all labels that are suitable to the review. "
                            "The output must be set of labels numbers separated by spaces"
                "There is also hierarchy: if you put one of labels 5,6,7,8,9,10,11 there must also be label 4. And if you put one of labels 8,9 there must be also label 7."
                "Example Review and Classification:"
                "Review: 'Bro, great, I want to add buildings to my city, but it requires premium, it wasn't like this in the past, it is like this now, it would be nice if it didn't require VIP, other than that, some modes do not work.'"
                "Expected Output: 1 4 7 8 11"

                 }

class GPT3:
    def __init__(self) -> None:

        openai.api_key = self.key

    def one_prompt(self, text_message, prompt=default_one_prompt):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens=256,
            temperature=0.4,  # Random of output.
            top_p=0.5,
            frequency_penalty=0,  # Common words. Higher value - less common words
            presence_penalty=0,  # Follow context. Higher value - more follow
            messages=[
                prompt,
                {"role": "user", "content": f"Review text is: {text_message}"},
            ]
        )
        return completion.choices[0].message.content, completion.usage

    def each_label_prompt(self, text_message, prompts):
        completions = []
        tokens = []
        for prompt in prompts:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                max_tokens=256,
                temperature=0.4,  # Random of output.
                top_p=0.5,
                frequency_penalty=0,  # Common words. Higher value - less common words
                presence_penalty=0,  # Follow context. Higher value - more follow
                messages=[
                    prompt,
                    {"role": "user", "content": f"Review text is: {text_message}"},
                ]
            )
            completions.append(completion.choices[0].message.content)
            tokens.append(completion.usage)

        totalUsage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        for token in tokens:
            totalUsage['prompt_tokens'] += token['prompt_tokens']
            totalUsage['completion_tokens'] += token['completion_tokens']
            totalUsage['total_tokens'] += token['total_tokens']

        return completions, totalUsage, tokens

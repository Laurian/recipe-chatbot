# https://gist.github.com/skylarbpayne/05268d32985bd2774cead74d1ea2d071
"""Script for generating queries for a recipe search engine.

This script can be used to generate synthetic queries for a recipe search engine.
Following best practices from Hamel and Shreya's AI Evals course, we:

- Generate a set of dimensions that can be used to generate queries; these are attributes that significantly change what the query is about or how it is written.
- For each set of attributes ("Dimensions") we generate a query that matches those attributes.

To ensure that the synthetic generation is better aligned, we first try handwriting the queries using the --manual flag.
This gives us labeled examples to use few shot in our synthetic generation.

Using the --verify-dims and --verify-queries flags, we can manually verify the dimensions and queries after generation so that we can remove any that are not useful.

This script uses [Mirascope](https://github.com/mirascope/mirascope) to generate the queries. I personally find it to provide an interface to LLMs that I like :).
Mirascope provides some helpful decorators to provide more structure to LLM calls. Instead of modeling the data as a conversation or as text in and text out,
we can define the structure of our input and the structure of our output data. I think it has a lot of nice features that make working with LLMs in the context of a
larger system much simpler (because everything becomes *just* a function).
"""

from mirascope import llm, prompt_template
from mirascope.core import FromCallArgs
from pydantic import BaseModel, Field
from typing import Literal, Annotated
import argparse
import csv


class Dimensions(BaseModel):
    cuisine: str | None = Field(default=None, description="The cuisine of the recipe")
    dietary_restriction: str | None = Field(default=None, description="The dietary restriction of the recipe")
    available_ingredients: str | None = Field(default=None, description="The ingredients the user has available")
    meal_type: str | None = Field(default=None, description="The meal type of the recipe")
    cooking_time: str | None = Field(default=None, description="The cooking time of the recipe")
    skill_level: str | None = Field(default=None, description="The skill level of the user")
    english_proficiency: Literal["native", "non-native"] | None = Field(default=None, description="The English proficiency of the user")


# the llm.call decorator let's us specify certain attributes about the llm call parameters
# the prompt_template decorator let's us specify the prompt template inline. It actually supports multiple messages with a special syntax (see documentation for more information)
# This let's us easily scaffold a prompt and provide the right inputs to it, all in a type safe way!
@llm.call(provider="openai", model="gpt-4o-mini", response_model=list[Dimensions])
@prompt_template("""SYSTEM:
<role>
You are an expert product manager. You know exactly how users think, what they want, and how they express themselves.
</role>
<instructions>
- Generate dimensions for possible recipe queries
- Try to populate at least 2 dimensions for each query
- Output {n} sets of dimensions
- Avoid outputting dimensions that have already been generated
</instructions>
<dimensions>           
- Specific cuisines (e.g., "Italian pasta dish", "Spicy Thai curry")
- Dietary restrictions (e.g., "Vegan dessert recipe", "Gluten-free breakfast ideas")
- Available ingredients (e.g., "What can I make with chicken, rice, and broccoli?")
- Meal types (e.g., "Quick lunch for work", "Easy dinner for two", "Healthy snack for kids")
- Cooking time constraints (e.g., "Recipe under 30 minutes")
- Skill levels (e.g., "Beginner-friendly baking recipe")
- English proficiency (e.g., "Native English speaker", "Non-native English speaker")
</dimensions>
<output_format>
[
    {{
        "cuisine": "Italian pasta dish",
        "dietary_restriction": "Gluten-free",
        "available_ingredients": "chicken, rice, broccoli",
        "meal_type": "Quick lunch for work",
    }},
    ...
]
</output_format>
<used_dimensions>
{used_dimensions_fmt}
</used_dimensions>
""")
def generate_dimensions(n: int = 10, used_dimensions: list[Dimensions] | None = None):
    # functions using the llm.call and prompt_template decorators don't need to even have a definition (can use ...).
    # however, we can use "computed" fields to help format some of our variables nicely
    used_dimensions_fmt = "\n".join([d.model_dump_json(indent=2) for d in used_dimensions])
    return {
        'computed_fields': {
            'used_dimensions_fmt': used_dimensions_fmt
        }
    }


class Query(BaseModel):
    # FromCallArgs is a cool little annotation that takes the argument directly from the call in generate_query. This way we package together the dimensions and the query generated
    # from it into a single object. See more: https://mirascope.com/docs/mirascope/learn/response_models/#fromcallargs
    dimensions: Annotated[Dimensions, FromCallArgs()] = Field(description="The dimensions of the query to guide the generation process")
    query: str = Field(description="The query that the user might use to search for a recipe that fits the dimensions")

    def xml(self) -> str:
        """Format the query in an XML format for easier parsing by LLMs"""
        return f"<example>\n<dimensions>{self.dimensions.model_dump_json(indent=2)}</dimensions>\n<query>{self.query}</query>\n</example>"


@llm.call(provider="openai", model="gpt-4o-mini", response_model=Query)
@prompt_template("""SYSTEM:
<role>
You are an expert product manager. You know exactly how users think, what they want, and how they express themselves.
</role>
<instructions>
- From the provided dimensions, generate a query that user might use to search for a recipe that fits those dimensions.
- Remember, users use simple language and don't always specify their intent (think ~6th grade level)
- Vary the phrasing of the queries you generate
- Use the provided dimensions to guide the generation process
</instructions>   
<examples>
{examples_fmt}
</examples>
<dimensions>
{dimensions_fmt}
</dimensions>
""")
def generate_query(dimensions: Dimensions, examples: list[Query] | None = None):
    examples_fmt = "\n".join([e.xml() for e in examples])
    return {
        'computed_fields': {
            'examples_fmt': examples_fmt,
            'dimensions_fmt': dimensions.model_dump_json(indent=2)
        }
    }

def get_query(dimensions: Dimensions) -> Query:
    """Get a query manually from user input"""
    print(dimensions.model_dump_json(indent=2))
    q = input("Query: ")
    return Query(query=q, dimensions=dimensions)


def should_keep(item: BaseModel) -> bool:
    """Ask user if they want to keep the item"""
    print(item.model_dump_json(indent=2))
    return input("Keep? (y/n): ") == "y"


def write_queries(queries: list[Query], output: str):
    """Write queries to a CSV file"""
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "query", "cuisine", "dietary_restriction", "available_ingredients", "meal_type", "cooking_time", "skill_level", "english_proficiency"])
        for i, q in enumerate(queries, 1):
            writer.writerow([i, q.query, q.dimensions.cuisine, q.dimensions.dietary_restriction, q.dimensions.available_ingredients, q.dimensions.meal_type, q.dimensions.cooking_time, q.dimensions.skill_level, q.dimensions.english_proficiency])


def load_queries(path: str | None = None) -> list[Query]:
    """Load examples from a CSV file"""
    if path is None:
        return []
    with open(path, "r") as f:
        reader = csv.reader(f)
        field_names = next(reader)
        return [Query(query=row[1], dimensions=Dimensions(**dict(zip(field_names, row[2:])))) for row in reader]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--verify-dims", action="store_true")
    parser.add_argument("--verify-queries", action="store_true")
    parser.add_argument("--examples", type=str, help="Path to a CSV file with examples of queries to use for few shot prompting")
    parser.add_argument("--output", type=str, help="Path to a CSV file to write the queries to")
    return parser.parse_args()


def main():
    args = parse_args()
    # These are previously generated example queries with their dimensions; this is helpful to provide the generator as few-shot examples :).
    examples = load_queries(args.examples)
    print(f"Loaded {len(examples)} examples")
    # here we call the llm function we created using mirascope which takes two arguments and returns a list of Dimensions objects.
    dimensions = generate_dimensions(args.n, [e.dimensions for e in examples])
    if args.verify_dims:
        dimensions = [d for d in dimensions if should_keep(d)]
    
    # here we selectively either ask the user for a query given the dimensions or we generate a query using the llm function we created using mirascope
    # depending on the --manual flag.
    queries = [get_query(d) if args.manual else generate_query(d, examples) for d in dimensions]
    if not args.manual and args.verify_queries:
        queries = [q for q in queries if should_keep(q)]
    print(f"Generated {len(queries)} queries")
    if args.output:
        write_queries(queries, args.output)
    for q in queries:
        print(q.dimensions.model_dump_json(indent=2))
        print(q.query)
        print()

if __name__ == "__main__":
    main()
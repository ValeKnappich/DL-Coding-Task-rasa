import json
import os
from itertools import combinations
import click


def load_and_transform(file_in):
  train = json.load(open(file_in, "r"))
  all_intents = list(set([instance["intent"] for _, instance in train.items()]))
  return all_intents, {
      intent: [dict(instance, id=id) for id, instance in train.items() if instance["intent"] == intent]
      for intent in all_intents
  }


def format_instance(instance):
  offset = 0
  result = instance["text"]
  for entity, (start, end) in sorted(instance["positions"].items(), key=lambda p: p[1][0]):
    start += offset
    end += offset + 1 
    result = result[:start] + "[" + result[start:end] + "](" + entity + ")" + result[end:]
    offset += 4 + len(entity)
  return result


def convert_to_yaml(file_out, train_by_intents, all_intents):
  with open(file_out, "w") as fp:
    fp.write("version: \"2.0\"\n")
    fp.write("nlu:\n")
    for intent in all_intents:
      fp.write(f"- intent: {intent}\n")
      fp.write("  examples: |\n")
      for instance in train_by_intents[intent]:
        fp.write(f"    - {format_instance(instance)}\n")


def check_problems(train_by_intents, clean):
    problem_names = ["Overlapping Entities", "Dot in Entity", 
                    "Trailing whitespace in Entity", 
                    "Entity value doesnt match Span"]
    problems = {problem: [] for problem in problem_names}
    problem_counts = {problem: 0 for problem in problem_names}
    cleaned = {}
    for intent, instances in train_by_intents.items():
        cleaned[intent] = []
        for instance in instances:
            problem = False
            if overlap(instance["positions"]):
                problems["Overlapping Entities"].append((instance["id"], None))
                problem_counts["Overlapping Entities"] += 1
                problem = True
            for (entity_name, entity), (start, stop) in zip(instance["slots"].items(), instance["positions"].values()):
                if "." in entity:
                    problems["Dot in Entity"].append((instance["id"], entity))
                    problem_counts["Dot in Entity"] += 1
                    problem = True
                elif entity.startswith(" "):
                    problems["Trailing whitespace in Entity"].append((instance["id"], entity))
                    problem_counts["Trailing whitespace in Entity"] += 1
                    problem = True
                elif entity != instance["text"][start:stop+1]:
                    problems["Entity value doesnt match Span"].append((instance["id"], entity))
                    problem_counts["Entity value doesnt match Span"] += 1
                    problem = True
            if not clean:
                # include in training data
                problem = False
            if not problem:
                cleaned[intent].append(instance)
    print(problem_counts)
    return cleaned, problems


def overlap(positions):
    for (start1, stop1), (start2, stop2) in combinations(positions.values(), 2):
        set1 = set(range(start1, stop1+1))
        set2 = set(range(start2, stop2+1))
        if set1.intersection(set2):
            return True
    return False


@click.command()
@click.option("-i", "--input", "file_in", default="data/train.json", help="Input file in json format")
@click.option("-o", "--output", "file_out", default="data/train.yml", help="Output file in yaml format")
@click.option("-c", "--check-only", default=False, is_flag=True, help="Only check for trailing whitespaces and dots inside of entities and not output yaml")
@click.option("-s", "--save-check", default=False, is_flag=True, help="Save check results to disk")
@click.option("--clean", default=False, is_flag=True, help="Exclude sample from dataset if a problem is found")
def main(file_in, file_out, check_only, save_check, clean):
    all_intents, train_by_intents = load_and_transform(file_in)
    train_by_intents, check_results = check_problems(train_by_intents, clean)
    if save_check:
        json.dump(check_results, open("data/check_results.json", "w"), indent=2)
    if not check_only:
        convert_to_yaml(file_out, train_by_intents, all_intents)


if __name__ == "__main__":
    main()
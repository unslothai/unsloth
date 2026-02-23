export type LearningRecipeDef = {
  id: string;
  title: string;
  description: string;
  filePath: string;
};

export const LEARNING_RECIPES: LearningRecipeDef[] = [
  {
    id: "structured-outputs-jinja",
    title: "Structured Outputs + Jinja Expressions",
    description: "Minimal schema + Jinja refs + if/else patterns.",
    filePath:
      "/src/features/data-recipes/learning-recipes/structured-outputs-jinja.json",
  },
];

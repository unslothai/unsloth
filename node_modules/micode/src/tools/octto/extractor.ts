// src/tools/octto/extractor.ts
// Utility functions for extracting answer summaries

import type {
  Answer,
  AskCodeAnswer,
  AskTextAnswer,
  ConfirmAnswer,
  EmojiReactAnswer,
  PickManyAnswer,
  PickOneAnswer,
  QuestionType,
  RankAnswer,
  RateAnswer,
  ReviewAnswer,
  ShowOptionsAnswer,
  SliderAnswer,
  ThumbsAnswer,
} from "../../octto/session";
import { QUESTIONS } from "../../octto/session";

const MAX_TEXT_LENGTH = 100;

function truncateText(text: string): string {
  return text.length > MAX_TEXT_LENGTH ? `${text.substring(0, MAX_TEXT_LENGTH)}...` : text;
}

export function extractAnswerSummary(type: QuestionType, answer: Answer): string {
  switch (type) {
    case QUESTIONS.PICK_ONE:
      return (answer as PickOneAnswer).selected;

    case QUESTIONS.PICK_MANY:
      return (answer as PickManyAnswer).selected.join(", ");

    case QUESTIONS.CONFIRM:
      return (answer as ConfirmAnswer).choice;

    case QUESTIONS.THUMBS:
      return (answer as ThumbsAnswer).choice;

    case QUESTIONS.EMOJI_REACT:
      return (answer as EmojiReactAnswer).emoji;

    case QUESTIONS.ASK_TEXT:
      return truncateText((answer as AskTextAnswer).text);

    case QUESTIONS.SLIDER:
      return String((answer as SliderAnswer).value);

    case QUESTIONS.RANK: {
      const rankAnswer = answer as RankAnswer;
      const sorted = [...rankAnswer.ranking].sort((a, b) => a.rank - b.rank);
      return sorted.map((r) => r.id).join(" → ");
    }

    case QUESTIONS.RATE: {
      const rateAnswer = answer as RateAnswer;
      const entries = Object.entries(rateAnswer.ratings);
      if (entries.length === 0) return "no ratings";
      const sorted = entries.sort((a, b) => b[1] - a[1]);
      return sorted
        .slice(0, 3)
        .map(([k, v]) => `${k}: ${v}`)
        .join(", ");
    }

    case QUESTIONS.ASK_CODE:
      return truncateText((answer as AskCodeAnswer).code);

    case QUESTIONS.ASK_IMAGE:
    case QUESTIONS.ASK_FILE:
      return "file(s) uploaded";

    case QUESTIONS.SHOW_DIFF:
    case QUESTIONS.SHOW_PLAN:
    case QUESTIONS.REVIEW_SECTION: {
      const reviewAnswer = answer as ReviewAnswer;
      return reviewAnswer.feedback
        ? `${reviewAnswer.decision}: ${truncateText(reviewAnswer.feedback)}`
        : reviewAnswer.decision;
    }

    case QUESTIONS.SHOW_OPTIONS: {
      const optAnswer = answer as ShowOptionsAnswer;
      return optAnswer.feedback ? `${optAnswer.selected}: ${truncateText(optAnswer.feedback)}` : optAnswer.selected;
    }

    default: {
      // Exhaustiveness check - if we get here, we missed a case
      const _exhaustive: never = type;
      return String(_exhaustive);
    }
  }
}

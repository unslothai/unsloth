// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type MessageTree = {
  readonly [key: string]: string | MessageTree;
};

export type DeepPartialMessageTree<T> = {
  readonly [K in keyof T]?: T[K] extends string
    ? string
    : T[K] extends MessageTree
      ? DeepPartialMessageTree<T[K]>
      : never;
};

type Join<Prefix extends string, Key extends string> =
  Prefix extends "" ? Key : `${Prefix}.${Key}`;

export type MessageKey<T, Prefix extends string = ""> = {
  [K in Extract<keyof T, string>]: T[K] extends string
    ? Join<Prefix, K>
    : T[K] extends MessageTree
      ? MessageKey<T[K], Join<Prefix, K>>
      : never;
}[Extract<keyof T, string>];

export type InterpolationValues = Record<
  string,
  string | number | boolean | null | undefined
>;

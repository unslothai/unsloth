// orderedmap 2.1.1 from npm (sha512-TvAWxi0nDe1j/rtMcWcIj94+Ffe6n7zhow33h40SKxmsmozs6dz/e+EajymfoFcHd7sxNn8yHM8839uixMOV6g==), MIT: see LICENSE. Imports point at the sibling vendored packages.
declare class OrderedMap<T = any> {
  private constructor(content: Array<string | T>)

  get(key: string): T | undefined

  update(key: string, value: T, newKey?: string): OrderedMap<T>

  remove(key: string): OrderedMap<T>

  addToStart(key: string, value: T): OrderedMap<T>

  addToEnd(key: string, value: T): OrderedMap<T>

  addBefore(place: string, key: string, value: T): OrderedMap<T>

  forEach(fn: (key: string, value: T) => any): void

  prepend(map: MapLike<T>): OrderedMap<T>

  append(map: MapLike<T>): OrderedMap<T>

  subtract(map: MapLike<T>): OrderedMap<T>

  toObject(): Record<string, T>;

  readonly size: number

  static from<T>(map: MapLike<T>): OrderedMap<T>
}

export type MapLike<T = any> = Record<string, T> | OrderedMap<T>

export default OrderedMap

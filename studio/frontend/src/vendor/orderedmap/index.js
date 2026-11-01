// orderedmap 2.1.1 from npm (sha512-TvAWxi0nDe1j/rtMcWcIj94+Ffe6n7zhow33h40SKxmsmozs6dz/e+EajymfoFcHd7sxNn8yHM8839uixMOV6g==), MIT: see LICENSE. Imports point at the sibling vendored packages.
// ::- Persistent data structure representing an ordered mapping from
// strings to values, with some convenient update methods.
function OrderedMap(content) {
  this.content = content;
}

OrderedMap.prototype = {
  constructor: OrderedMap,

  find: function(key) {
    for (var i = 0; i < this.content.length; i += 2)
      if (this.content[i] === key) return i
    return -1
  },

  // :: (string) → ?any
  // Retrieve the value stored under `key`, or return undefined when
  // no such key exists.
  get: function(key) {
    var found = this.find(key);
    return found == -1 ? undefined : this.content[found + 1]
  },

  // :: (string, any, ?string) → OrderedMap
  // Create a new map by replacing the value of `key` with a new
  // value, or adding a binding to the end of the map. If `newKey` is
  // given, the key of the binding will be replaced with that key.
  update: function(key, value, newKey) {
    var self = newKey && newKey != key ? this.remove(newKey) : this;
    var found = self.find(key), content = self.content.slice();
    if (found == -1) {
      content.push(newKey || key, value);
    } else {
      content[found + 1] = value;
      if (newKey) content[found] = newKey;
    }
    return new OrderedMap(content)
  },

  // :: (string) → OrderedMap
  // Return a map with the given key removed, if it existed.
  remove: function(key) {
    var found = this.find(key);
    if (found == -1) return this
    var content = this.content.slice();
    content.splice(found, 2);
    return new OrderedMap(content)
  },

  // :: (string, any) → OrderedMap
  // Add a new key to the start of the map.
  addToStart: function(key, value) {
    return new OrderedMap([key, value].concat(this.remove(key).content))
  },

  // :: (string, any) → OrderedMap
  // Add a new key to the end of the map.
  addToEnd: function(key, value) {
    var content = this.remove(key).content.slice();
    content.push(key, value);
    return new OrderedMap(content)
  },

  // :: (string, string, any) → OrderedMap
  // Add a key after the given key. If `place` is not found, the new
  // key is added to the end.
  addBefore: function(place, key, value) {
    var without = this.remove(key), content = without.content.slice();
    var found = without.find(place);
    content.splice(found == -1 ? content.length : found, 0, key, value);
    return new OrderedMap(content)
  },

  // :: ((key: string, value: any))
  // Call the given function for each key/value pair in the map, in
  // order.
  forEach: function(f) {
    for (var i = 0; i < this.content.length; i += 2)
      f(this.content[i], this.content[i + 1]);
  },

  // :: (union<Object, OrderedMap>) → OrderedMap
  // Create a new map by prepending the keys in this map that don't
  // appear in `map` before the keys in `map`.
  prepend: function(map) {
    map = OrderedMap.from(map);
    if (!map.size) return this
    return new OrderedMap(map.content.concat(this.subtract(map).content))
  },

  // :: (union<Object, OrderedMap>) → OrderedMap
  // Create a new map by appending the keys in this map that don't
  // appear in `map` after the keys in `map`.
  append: function(map) {
    map = OrderedMap.from(map);
    if (!map.size) return this
    return new OrderedMap(this.subtract(map).content.concat(map.content))
  },

  // :: (union<Object, OrderedMap>) → OrderedMap
  // Create a map containing all the keys in this map that don't
  // appear in `map`.
  subtract: function(map) {
    var result = this;
    map = OrderedMap.from(map);
    for (var i = 0; i < map.content.length; i += 2)
      result = result.remove(map.content[i]);
    return result
  },

  // :: () → Object
  // Turn ordered map into a plain object.
  toObject: function() {
    var result = {};
    this.forEach(function(key, value) { result[key] = value; });
    return result
  },

  // :: number
  // The amount of keys in this map.
  get size() {
    return this.content.length >> 1
  }
};

// :: (?union<Object, OrderedMap>) → OrderedMap
// Return a map with the given content. If null, create an empty
// map. If given an ordered map, return that map itself. If given an
// object, create a map from the object's properties.
OrderedMap.from = function(value) {
  if (value instanceof OrderedMap) return value
  var content = [];
  if (value) for (var prop in value) content.push(prop, value[prop]);
  return new OrderedMap(content)
};

export default OrderedMap;

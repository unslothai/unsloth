export function resolve(specifier, context, next) {
  if (specifier.endsWith('/types/runtime')) return next(specifier + '.ts', context);
  return next(specifier, context);
}

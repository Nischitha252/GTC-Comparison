// IconUtils.ts
export type Size = 'small' | 'medium' | 'large' | 'extraLarge';

export const sizeDimensions: Record<Size, { width: number; height: number }> = {
  small: { width: 16, height: 16 },
  medium: { width: 24, height: 24 },
  large: { width: 32, height: 32 },
  extraLarge: { width: 48, height: 48 }
};

export const getSizeDimensions = (size: Size) => sizeDimensions[size];

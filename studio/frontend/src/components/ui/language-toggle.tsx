import { useTranslation } from 'react-i18next';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { GlobeIcon } from '@hugeicons/core-free-icons';
import { HugeiconsIcon } from '@hugeicons/react';
import type { ReactElement } from 'react';

export function LanguageToggle(): ReactElement {
  const { i18n } = useTranslation();

  const currentLang = i18n.language || 'en-US';
  
  const languages = [
    { value: 'en-US', label: 'English' },
    { value: 'zh-CN', label: '简体中文' },
  ];

  const handleLanguageChange = (value: string) => {
    i18n.changeLanguage(value);
  };

  return (
    <Select value={currentLang} onValueChange={handleLanguageChange}>
      <SelectTrigger className="h-9 w-[110px] rounded-md border border-border bg-transparent px-3 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground">
        <div className="flex items-center gap-1.5">
          <HugeiconsIcon icon={GlobeIcon} className="size-4" />
          <SelectValue placeholder="Language" />
        </div>
      </SelectTrigger>
      <SelectContent>
        {languages.map((lang) => (
          <SelectItem key={lang.value} value={lang.value}>
            {lang.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

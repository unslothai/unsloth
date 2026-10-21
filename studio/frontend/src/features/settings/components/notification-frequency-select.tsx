// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  NOTIFICATION_FREQUENCIES,
  type NotificationChannel,
  type NotificationFrequency,
  setNotificationFrequency,
  useNotificationFrequency,
} from "@/hooks/use-notification-frequency";
import { useT } from "@/i18n";

const LABEL_KEYS = {
  always: "settings.general.notifications.frequency.always",
  daily: "settings.general.notifications.frequency.daily",
  weekly: "settings.general.notifications.frequency.weekly",
  biweekly: "settings.general.notifications.frequency.biweekly",
  monthly: "settings.general.notifications.frequency.monthly",
  off: "settings.general.notifications.frequency.off",
} as const satisfies Record<NotificationFrequency, string>;

export function NotificationFrequencySelect({
  channel,
  label,
}: {
  channel: NotificationChannel;
  label: string;
}) {
  const t = useT();
  const frequency = useNotificationFrequency(channel);
  return (
    <Select
      value={frequency}
      onValueChange={(value) =>
        setNotificationFrequency(channel, value as NotificationFrequency)
      }
    >
      <SelectTrigger aria-label={label} className="w-40" size="sm">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {NOTIFICATION_FREQUENCIES.map((value) => (
          <SelectItem key={value} value={value}>
            {t(LABEL_KEYS[value])}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
